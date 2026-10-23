"""Vectorised active-search ('gaze') environment over cached glimpse embeddings (pure numpy).

State   : current zoom level l, cell (row, col), step t, visited map.
Observe : fovea embedding of the current glimpse; a low-res peripheral map (PCA of mid-level
          embeddings + visited mask + current footprint); level one-hot, position, time.
Actions : 0 up, 1 down, 2 left, 3 right, 4-7 zoom into quadrant (TL, TR, BL, BR), 8 zoom out, 9 STOP,
          10 + k: SACCADE - jump the fovea to cell k of the peripheral grid (at the peripheral level).
          STOP is only legal at the finest level; illegal actions are masked. The saccade head of the
          policy is a per-cell pointer over the peripheral map, i.e. a learned fixation-priority map.
Reward  : -step_cost per step; on STOP the spec's terminal score of the fixated cell; on timeout a
          fixed penalty. Optional potential-based shaping (Ng et al. 1999) with the spec score.
"""
from __future__ import annotations

import numpy as np

N_BASE = 10
STOP = 9


class GlimpseEnv:
    def __init__(self, emb, periph, terminal, potential, succ_L, levels, env_cfg, text=None):
        self.emb = emb                    # list[L] of [n_img, N_l, D] float16
        self.periph = periph              # [n_img, C, Gp, Gp] float32
        self.terminal = terminal          # [n_img, N_L] float32  terminal reward on STOP
        self.potential = potential        # list[L] of [n_img, N_l] float32 (shaping potential)
        self.succ_L = succ_L              # [n_img, N_L] bool     ground truth, for metrics only
        self.text = text                  # [n_img, D] target-text embedding per image
        self.levels = np.asarray(levels)
        self.L = len(levels)
        self.Gp = periph.shape[-1]
        if self.Gp not in levels:
            raise ValueError("peripheral grid must equal one of the zoom levels")
        self.lp = int(list(levels).index(self.Gp))
        self.n_actions = N_BASE + self.Gp * self.Gp
        self.T = int(env_cfg["horizon"])
        self.step_cost = float(env_cfg["step_cost"])
        self.timeout_penalty = float(env_cfg["timeout_penalty"])
        self.shaping = bool(env_cfg["shaping"])
        self.gamma = float(env_cfg["gamma"])

    # ------------------------------------------------------------------ episode control
    def reset(self, img: np.ndarray, start_cell: np.ndarray):
        n = len(img)
        self.n = n
        self.img = img.astype(np.int64)
        self.l = np.zeros(n, np.int64)
        G0 = self.levels[0]
        self.r, self.c = start_cell // G0, start_cell % G0
        self.t = np.zeros(n, np.int64)
        self.done = np.zeros(n, bool)
        self.visited = np.zeros((n, self.Gp, self.Gp), np.float32)
        self.visited = np.maximum(self.visited, self._footprint())
        self.ep_return = np.zeros(n, np.float32)
        self.info = {
            "success": np.zeros(n, bool), "timeout": np.zeros(n, bool),
            "steps": np.zeros(n, np.int64), "stop_cell": np.full(n, -1, np.int64),
            "first_success_step": np.full(n, -1, np.int64),
        }
        self.trace = [self._fix()]       # fixation history (level, row, col) for figures

    def _fix(self):
        return np.stack([self.l, self.r, self.c], 1).copy()

    def _cell(self):
        return self.r * self.levels[self.l] + self.c

    def _footprint(self) -> np.ndarray:
        G = self.levels[self.l]
        Gp = self.Gp
        lo_r = self.r * Gp // G
        hi_r = np.maximum(lo_r, ((self.r + 1) * Gp - 1) // G)
        lo_c = self.c * Gp // G
        hi_c = np.maximum(lo_c, ((self.c + 1) * Gp - 1) // G)
        ar = np.arange(Gp)
        fr = (ar[None] >= lo_r[:, None]) & (ar[None] <= hi_r[:, None])
        fc = (ar[None] >= lo_c[:, None]) & (ar[None] <= hi_c[:, None])
        return (fr[:, :, None] & fc[:, None, :]).astype(np.float32)

    def _phi(self) -> np.ndarray:
        cell = self._cell()
        out = np.zeros(self.n, np.float32)
        for l in range(self.L):
            m = self.l == l
            if m.any():
                out[m] = self.potential[l][self.img[m], cell[m]]
        return out

    # ------------------------------------------------------------------ observation
    def action_mask(self) -> np.ndarray:
        G = self.levels[self.l]
        m = np.zeros((self.n, self.n_actions), bool)
        m[:, 0] = self.r > 0
        m[:, 1] = self.r < G - 1
        m[:, 2] = self.c > 0
        m[:, 3] = self.c < G - 1
        m[:, 4:8] = (self.l < self.L - 1)[:, None]
        m[:, 8] = self.l > 0
        m[:, STOP] = self.l == self.L - 1
        m[:, N_BASE:] = True
        m[self.done] = False
        m[self.done, 0] = True            # dummy legal action for finished envs (ignored in loss)
        return m

    def obs(self) -> dict:
        cell = self._cell()
        D = self.emb[0].shape[-1]
        fov = np.zeros((self.n, D), np.float32)
        for l in range(self.L):
            m = self.l == l
            if m.any():
                fov[m] = self.emb[l][self.img[m], cell[m]]
        if self.text is not None:
            fov = np.concatenate([fov, fov * self.text[self.img]], 1)
        else:
            fov = np.concatenate([fov, np.zeros_like(fov)], 1)
        per = np.concatenate([self.periph[self.img], self.visited[:, None], self._footprint()[:, None]], 1)
        per = per.transpose(0, 2, 3, 1)                 # cell-major: [n, Gp, Gp, C+2] -> flattened below
        meta = np.zeros((self.n, self.L + 3), np.float32)
        meta[np.arange(self.n), self.l] = 1.0
        Gm1 = np.maximum(self.levels[self.l] - 1, 1)
        meta[:, self.L] = self.r / Gm1
        meta[:, self.L + 1] = self.c / Gm1
        meta[:, self.L + 2] = self.t / self.T
        return {"fov": fov, "per": per.reshape(self.n, -1).astype(np.float32), "meta": meta,
                "mask": self.action_mask()}

    # ------------------------------------------------------------------ dynamics
    def step(self, a: np.ndarray):
        act = ~self.done
        a = np.where(act, a, -1)
        phi0 = self._phi() if self.shaping else None
        G = self.levels[self.l]

        self.r = np.where(a == 0, np.maximum(self.r - 1, 0), self.r)
        self.r = np.where(a == 1, np.minimum(self.r + 1, G - 1), self.r)
        self.c = np.where(a == 2, np.maximum(self.c - 1, 0), self.c)
        self.c = np.where(a == 3, np.minimum(self.c + 1, G - 1), self.c)
        zin = (a >= 4) & (a <= 7) & (self.l < self.L - 1)
        q = np.clip(a - 4, 0, 3)
        self.r = np.where(zin, 2 * self.r + q // 2, self.r)
        self.c = np.where(zin, 2 * self.c + q % 2, self.c)
        self.l = np.where(zin, self.l + 1, self.l)
        zout = (a == 8) & (self.l > 0)
        self.r = np.where(zout, self.r // 2, self.r)
        self.c = np.where(zout, self.c // 2, self.c)
        self.l = np.where(zout, self.l - 1, self.l)

        sac = act & (a >= N_BASE)
        k = np.clip(a - N_BASE, 0, None)
        self.r = np.where(sac, k // self.Gp, self.r)
        self.c = np.where(sac, k % self.Gp, self.c)
        self.l = np.where(sac, self.lp, self.l)

        stop = act & (a == STOP) & (self.l == self.L - 1)
        self.t = np.where(act, self.t + 1, self.t)
        reward = np.where(act, -self.step_cost, 0.0).astype(np.float32)
        cell = self._cell()
        at_fine = self.l == self.L - 1
        succ_here = np.zeros(self.n, bool)
        succ_here[at_fine] = self.succ_L[self.img[at_fine], cell[at_fine]]
        first = act & succ_here & (self.info["first_success_step"] < 0)
        self.info["first_success_step"][first] = self.t[first]

        if stop.any():
            reward[stop] += self.terminal[self.img[stop], cell[stop]]
        timeout = act & ~stop & (self.t >= self.T)
        reward[timeout] += self.timeout_penalty
        newly_done = stop | timeout

        if self.shaping:
            phi1 = self._phi()
            phi1[newly_done] = 0.0                                  # terminal potential = 0
            reward += np.where(act, self.gamma * phi1 - phi0, 0.0).astype(np.float32)

        self.info["success"][stop] = succ_here[stop]
        self.info["timeout"][timeout] = True
        self.info["steps"][newly_done] = self.t[newly_done]
        self.info["stop_cell"][stop] = cell[stop]
        self.done |= newly_done
        live = act & ~newly_done
        if live.any():
            self.visited[live] = np.maximum(self.visited[live], self._footprint()[live])
        self.ep_return += reward
        self.trace.append(self._fix())
        return reward, newly_done, act
