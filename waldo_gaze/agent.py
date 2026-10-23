"""Actor-critic gaze policy and PPO (clipped surrogate, GAE)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .env import N_BASE


class GazePolicy(nn.Module):
    """Actor-critic with two action heads:
    base head    : move / zoom / stop logits from the global state;
    saccade head : a pointer over peripheral cells, logit_k = f(cell_k features, global context).
    The saccade logits form a learned fixation-priority ('gaze') map over the scene."""

    def __init__(self, fov_dim: int, n_cells: int, cell_dim: int, meta_dim: int, hidden: int,
                 sim_prior: float = 0.0, sim_index: int | None = None):
        super().__init__()
        # residual prior: saccades start where CLIP text-similarity is high; RL learns a correction
        self.sim_prior, self.sim_index = float(sim_prior), sim_index
        self.n_cells, self.cell_dim = n_cells, cell_dim
        self.fov = nn.Sequential(nn.Linear(fov_dim, hidden), nn.ReLU())
        # global periphery summary = mean+max pooled cells (NOT the flattened map: with ~30 training
        # images a flattened map lets the policy fingerprint and memorise each image)
        self.per = nn.Sequential(nn.Linear(2 * cell_dim, hidden), nn.ReLU())
        self.meta = nn.Sequential(nn.Linear(meta_dim, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(2 * hidden + 64, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, N_BASE)
        self.cell = nn.Linear(cell_dim, 64)
        self.ctx = nn.Linear(hidden, 64)
        self.sac = nn.Linear(64, 1)
        self.v = nn.Linear(hidden, 1)

    def forward(self, fov, per, meta, mask):
        n = fov.shape[0]
        cells = per.reshape(n, self.n_cells, self.cell_dim)
        pooled = torch.cat([cells.mean(1), cells.amax(1)], -1)
        h = self.trunk(torch.cat([self.fov(fov), self.per(pooled), self.meta(meta)], -1))
        ch = torch.relu(self.cell(cells) + self.ctx(h).reshape(n, 1, 64))
        # fixed prior: the whole saccade head starts with the mass of ONE base action, so a fresh
        # policy still zooms and stops instead of saccading until every episode times out
        sac = self.sac(ch).squeeze(-1)
        if self.sim_prior and self.sim_index is not None:
            # redistribute (not increase) the saccade head's prior mass toward high-similarity cells
            pz = self.sim_prior * cells[:, :, self.sim_index].detach()
            sac = sac + pz - torch.logsumexp(pz, -1, keepdim=True)
        else:
            sac = sac - math.log(self.n_cells)
        logits = torch.cat([self.pi(h), sac], -1).masked_fill(~mask, -1e9)
        return logits, self.v(h).squeeze(-1)


def _to(o: dict, device) -> dict:
    return {k: torch.from_numpy(v).to(device) for k, v in o.items()}


def rollout(env, policy, img, start, device, greedy: bool, rng: torch.Generator | None = None):
    """Run one synchronised batch of episodes to completion. Returns per-step tensors."""
    env.reset(img, start)
    steps = []
    for _ in range(env.T):
        o = _to(env.obs(), device)
        with torch.no_grad():
            logits, v = policy(o["fov"], o["per"], o["meta"], o["mask"])
            if greedy:
                a = logits.argmax(-1)
            else:
                u = torch.rand(logits.shape, generator=rng, device=device).clamp(1e-6, 1 - 1e-6)
                a = (logits - torch.log(-torch.log(u))).argmax(-1)       # Gumbel-max sampling
            logp = torch.log_softmax(logits, -1).gather(-1, a[:, None]).squeeze(-1)
        r, newly_done, active = env.step(a.cpu().numpy())
        steps.append({"obs": o, "a": a, "logp": logp, "v": v,
                      "img": torch.from_numpy(env.img).to(device),
                      "r": torch.from_numpy(r).to(device),
                      "done": torch.from_numpy(newly_done).to(device),
                      "active": torch.from_numpy(active).to(device)})
        if env.done.all():
            break
    return steps


def gae(steps, gamma: float, lam: float):
    n = len(steps)
    adv = [None] * n
    last = torch.zeros_like(steps[0]["v"])
    for k in reversed(range(n)):
        nonterm = 1.0 - steps[k]["done"].float()
        next_v = steps[k + 1]["v"] if k + 1 < n else torch.zeros_like(steps[k]["v"])
        delta = steps[k]["r"] + gamma * next_v * nonterm - steps[k]["v"]
        last = delta + gamma * lam * nonterm * last
        adv[k] = last * steps[k]["active"].float()
    return adv


def ppo_update(policy, opt, steps, cfg: dict, gamma: float, gen: torch.Generator, device,
               gaze=None, gaze_valid=None, gaze_coef: float = 0.0) -> dict:
    """Clipped PPO. If gaze_coef > 0, adds a coverage-style gaze loss (cf. CGL, Saran et al. 2021):
    cross-entropy between the human fixation map of the image and the policy's saccade distribution,
    i.e. the agent is penalised for putting little saccade mass where people looked. It touches only
    the saccade head's distribution, never the reward."""
    adv = gae(steps, gamma, cfg["gae_lambda"])
    act = torch.cat([s["active"] for s in steps])
    keep = torch.nonzero(act).squeeze(-1)
    cat = lambda key: torch.cat([s["obs"][key] for s in steps])[keep]
    fov, per, meta, mask = cat("fov"), cat("per"), cat("meta"), cat("mask")
    a = torch.cat([s["a"] for s in steps])[keep]
    img = torch.cat([s["img"] for s in steps])[keep]
    use_gaze = gaze is not None and gaze_coef > 0
    old_logp = torch.cat([s["logp"] for s in steps])[keep]
    A = torch.cat(adv)[keep]
    ret = A + torch.cat([s["v"] for s in steps])[keep]
    A = (A - A.mean()) / (A.std() + 1e-8)
    N = A.shape[0]
    stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "gaze": 0.0, "n": 0}
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(N, generator=gen, device=device)
        for s in range(0, N, cfg["minibatch"]):
            mb = perm[s:s + cfg["minibatch"]]
            logits, v = policy(fov[mb], per[mb], meta[mb], mask[mb])
            logp_all = torch.log_softmax(logits, -1)
            logp = logp_all.gather(-1, a[mb][:, None]).squeeze(-1)
            ratio = torch.exp(logp - old_logp[mb])
            pg = -torch.min(ratio * A[mb], torch.clamp(ratio, 1 - cfg["clip"], 1 + cfg["clip"]) * A[mb]).mean()
            probs = torch.exp(logp_all)
            ent = -(probs * logp_all.masked_fill(~mask[mb], 0.0)).sum(-1).mean()
            vf = ((v - ret[mb]) ** 2).mean()
            loss = pg + cfg["vf_coef"] * vf - cfg["ent_coef"] * ent
            if use_gaze:
                g = gaze[img[mb]]                                            # [B, Gp*Gp]
                w = gaze_valid[img[mb]].float()
                log_sac = torch.log_softmax(logits[:, N_BASE:], -1)
                ce = -(g * log_sac).sum(-1)
                gl = (w * ce).sum() * (1.0 / max(float(w.sum()), 1.0))
                loss = loss + gaze_coef * gl
                stats["gaze"] += float(gl)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
            opt.step()
            stats["pg"] += float(pg)
            stats["vf"] += float(vf)
            stats["ent"] += float(ent)
            stats["n"] += 1
    return {k: v / max(stats["n"], 1) for k, v in stats.items() if k != "n"}
