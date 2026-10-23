"""Stage 3: aggregate runs -> tables (markdown + LaTeX) and figures, with honest uncertainty.

Unit of analysis = image. Each image is a test image in exactly one fold; its metric is averaged
over seeds, then we bootstrap over images (95% percentile CI). Paired comparisons use the
per-image difference (paired bootstrap CI + Wilcoxon signed-rank). Seed std is reported too.

  python -m waldo_gaze.analyze --config configs/waldo.yaml
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from .data import load_all_cache, load_config
from .rewards import zscore

N_BOOT = 10_000
METRICS = ("success", "hacked", "distractor", "near_miss", "timeout", "found_not_stopped", "steps")
# categorical slots in fixed order (validated: CVD + normal-vision separation pass; low-contrast
# slots carry markers + direct labels as secondary encoding)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def boot_ci(x: np.ndarray, rng) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    bs = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load_runs(res_dir: str) -> list[dict]:
    runs = []
    for p in sorted(glob.glob(os.path.join(res_dir, "runs", "*.json"))):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def per_image_table(runs, split_key="test"):
    """spec -> metric -> {image_id: mean over seeds};   spec -> seed -> metric -> mean over images"""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    by_seed = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in runs:
        for row in r[split_key]:
            for m in METRICS + ("gaze_cc",):
                if row.get(m) is None:
                    continue
                acc[r["spec"]][m][row["image_id"]].append(row[m])
                by_seed[r["spec"]][r["seed"]][m].append(row[m])
    img = {s: {m: {i: float(np.mean(v)) for i, v in d.items()} for m, d in ms.items()} for s, ms in acc.items()}
    seed = {s: {sd: {m: float(np.mean(v)) for m, v in ms.items()} for sd, ms in d.items()} for s, d in by_seed.items()}
    return img, seed


def load_static(res_dir: str):
    out = defaultdict(dict)                    # spec -> image_id -> row
    kappas = defaultdict(list)
    for p in glob.glob(os.path.join(res_dir, "static", "*.json")):
        with open(p) as f:
            d = json.load(f)
        spec = d["info"]["spec"]
        if "kappa" in d["info"]:
            kappas[spec].append(d["info"]["kappa"])
        for row in d["test"]:
            out[spec][row["image_id"]] = row
    return out, kappas


def prompt_diagnostic(cfg, rng) -> list[dict]:
    """Every single prompt, no learning, no folds: how often is the target the reward's argmax?
    (the honest version of a query-phrasing table: all images, CIs, nothing selected on test)"""
    data = load_all_cache(cfg)
    base = np.array([i for i, k in enumerate(data["ids"]) if data["aug_of"][k] is None])
    E = data["emb"][-1][base].astype(np.float32)
    S = data["succ"][-1][base]
    rows = []
    for name, vec in data["text"].items():
        z = zscore(np.einsum("ind,id->in", E, vec[base]), axis=1)     # each image vs its own target text
        top1 = np.array([S[i][np.argmax(z[i])] for i in range(len(z))], float)
        pct = np.array([np.mean(z[i][~S[i]] < z[i][S[i]].max()) for i in range(len(z))])
        nb = np.array([np.sum(z[i][~S[i]] > z[i][S[i]].max()) for i in range(len(z))])
        rows.append({"prompt": name, "text": cfg["prompts"].get(name, ""), "top1": boot_ci(top1, rng),
                     "gt_pct": boot_ci(pct, rng), "n_better_median": float(np.median(nb))})
    return sorted(rows, key=lambda r: -r["top1"][0])


def fmt(ci, pct=True, digits=1):
    m, lo, hi = ci
    if np.isnan(m):
        return "-"
    k = 100 if pct else 1
    return f"{m * k:.{digits}f} [{lo * k:.{digits}f}, {hi * k:.{digits}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--budget_specs", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    res_dir = cfg["results_dir"]
    out_dir = os.path.join(res_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    runs = load_runs(res_dir)
    if not runs:
        raise SystemExit("no runs found")
    specs = [s for s in cfg["specs"] if any(r["spec"] == s for r in runs)]
    img, seed = per_image_table(runs, "test")
    img_tr, _ = per_image_table(runs, "train")
    static, kappas = load_static(res_dir)
    with open(os.path.join(res_dir, "splits.json")) as f:
        n_images = sum(len(s["test"]) for s in json.load(f))

    # ---------------- completeness check (never silently report a partial grid)
    expected = len(cfg["protocol"]["seeds"]) * cfg["protocol"]["n_folds"]
    lines = ["# Results", "", f"Images: {n_images}. Unit = image (seed-averaged); 95% bootstrap CI over images.", ""]
    incomplete = []
    for s in specs:
        n = sum(r["spec"] == s for r in runs)
        cov = len(img[s]["success"])
        if n < expected or cov < n_images:
            incomplete.append(f"{s}: {n}/{expected} runs, {cov}/{n_images} images")
    if incomplete:
        lines += ["**WARNING - incomplete grid, numbers below are NOT final:**", *[f"- {x}" for x in incomplete], ""]

    # ---------------- main table
    summary = {}
    lines += ["## Main table (held-out images, greedy policy from all level-0 starts)", "",
              "| reward | RL success | hacked | distractor | near-miss | timeout | steps | train success | "
              "static top-1 | coarse-to-fine | seed std | kappa |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r"Reward & RL success $\uparrow$ & Hacked $\downarrow$ & Distractor $\downarrow$ & Train succ. & Static top-1 & C2F \\",
           r"\midrule"]
    for s in specs:
        ids = sorted(img[s]["success"])
        ci = {m: boot_ci([img[s][m][i] for i in ids], rng) for m in METRICS}
        tr_ids = sorted(img_tr[s]["success"])
        tr_ci = boot_ci([img_tr[s]["success"][i] for i in tr_ids], rng)
        st = static.get(s, {})
        top1 = boot_ci([float(st[i]["top1"]) for i in ids if i in st], rng)
        c2f = boot_ci([float(st[i]["c2f_success"]) for i in ids if i in st], rng)
        sstd = float(np.std([v["success"] for v in seed[s].values()])) if len(seed[s]) > 1 else float("nan")
        kap = ",".join(f"{k:g}" for k in sorted(set(kappas.get(s, [])))) or "-"
        summary[s] = {"metrics": ci, "train_success": tr_ci, "static_top1": top1, "c2f": c2f,
                      "seed_std": sstd, "kappa": kappas.get(s, []), "n_images": len(ids)}
        lines.append(f"| {s} | {fmt(ci['success'])} | {fmt(ci['hacked'])} | {fmt(ci['distractor'])} | "
                     f"{fmt(ci['near_miss'])} | {fmt(ci['timeout'])} | {ci['steps'][0]:.1f} | {fmt(tr_ci)} | "
                     f"{fmt(top1)} | {fmt(c2f)} | {sstd * 100:.1f} | {kap} |")
        tex_name = s.replace("_", "\\_")
        tex.append(f"{tex_name} & {ci['success'][0] * 100:.1f} $\\pm$ "
                   f"{(ci['success'][2] - ci['success'][1]) * 50:.1f} & {ci['hacked'][0] * 100:.1f} & "
                   f"{ci['distractor'][0] * 100:.1f} & {tr_ci[0] * 100:.1f} & {top1[0] * 100:.1f} & {c2f[0] * 100:.1f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    lines += ["", "hacked = stopped on a non-target glimpse that the TRAINING reward scores >= the best target glimpse.",
              "distractor = wrong stop >1 cell from the target; near-miss = wrong stop adjacent to it.", ""]

    # ---------------- paired comparisons
    lines += ["## Paired comparisons (A - B, held-out success, per image)", "",
              "| A | B | mean diff (pp) | 95% CI | Wilcoxon p | A>B / A<B / tie |", "|---|---|---|---|---|---|"]
    from scipy.stats import wilcoxon
    paired = []
    for a, b in cfg.get("compare", []):
        if a not in img or b not in img:
            continue
        ids = sorted(set(img[a]["success"]) & set(img[b]["success"]))
        d = np.array([img[a]["success"][i] - img[b]["success"][i] for i in ids])
        ci = boot_ci(d, rng)
        p = float(wilcoxon(d, zero_method="zsplit").pvalue) if np.any(d != 0) else 1.0
        paired.append({"a": a, "b": b, "diff": ci, "p": p, "n": len(ids)})
        lines.append(f"| {a} | {b} | {ci[0] * 100:+.1f} | [{ci[1] * 100:+.1f}, {ci[2] * 100:+.1f}] | {p:.3g} | "
                     f"{int((d > 0).sum())} / {int((d < 0).sum())} / {int((d == 0).sum())} |")
    lines += ["", "No multiple-comparison correction is applied; with "
              f"{len(paired)} comparisons, treat p > {0.05 / max(len(paired), 1):.3f} (Bonferroni) as inconclusive.", ""]

    # ---------------- prompt diagnostic
    pdiag = prompt_diagnostic(cfg, rng)
    lines += ["## Single-prompt diagnostic (no RL, all images)", "",
              "| prompt | top-1 (target is argmax) | target percentile | median # distractors above target |",
              "|---|---|---|---|"]
    for r in pdiag:
        lines.append(f"| {r['prompt']} | {fmt(r['top1'])} | {fmt(r['gt_pct'])} | {r['n_better_median']:.0f} |")

    with open(os.path.join(out_dir, "table_main.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    # ---------------- human-gaze agreement (datasets with fixations, e.g. COCO-Search18)
    gaze_rows = []
    if any("gaze_cc" in img[s] and img[s]["gaze_cc"] for s in specs):
        cb = center_bias_cc(cfg)
        lines += ["", "## Agreement with human search fixations (held-out images)", "",
                  "Pearson CC between the agent's pooled fixation map and the human fixation map on the "
                  "peripheral grid. Reference: a centre-bias map scores "
                  f"{fmt(boot_ci(list(cb.values()), rng), pct=False, digits=3)}.", "",
                  "| reward | gaze CC | success |", "|---|---|---|"]
        for s in specs:
            g = img[s].get("gaze_cc", {})
            if not g:
                continue
            ci = boot_ci([g[i] for i in sorted(g)], rng)
            gaze_rows.append({"spec": s, "cc": ci})
            lines.append(f"| {s} | {fmt(ci, pct=False, digits=3)} | {fmt(summary[s]['metrics']['success'])} |")

    # ---------------- dose-response by an image-level factor (e.g. number of partial distractors)
    group_rows = []
    gb = cfg.get("analysis", {}).get("group_by")
    if gb:
        meta = load_image_meta(cfg)
        levels_ = sorted({m.get(gb) for m in meta.values() if m.get(gb) is not None})
        lines += ["", f"## Breakdown by `{gb}` (held-out success / hacked, %)", "",
                  "| reward | " + " | ".join(f"{gb}={v}" for v in levels_) + " |",
                  "|---|" + "---|" * len(levels_)]
        for s in specs:
            cells = []
            for v in levels_:
                ids = [i for i in img[s]["success"] if meta.get(i, {}).get(gb) == v]
                su = boot_ci([img[s]["success"][i] for i in ids], rng)
                ha = boot_ci([img[s]["hacked"][i] for i in ids], rng)
                group_rows.append({"spec": s, "level": v, "success": su, "hacked": ha, "n": len(ids)})
                cells.append(f"{su[0] * 100:.1f} / {ha[0] * 100:.1f} (n={len(ids)})")
            lines.append(f"| {s} | " + " | ".join(cells) + " |")

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"summary": summary, "paired": paired, "prompts": pdiag, "incomplete": incomplete,
                   "gaze": gaze_rows, "groups": group_rows}, f, indent=1)

    budget = args.budget_specs or cfg.get("analysis", {}).get("budget_specs", ["oracle", "clip_full"])
    make_figures(specs, summary, runs, budget, cfg, out_dir, group_rows, gb)
    print("\n".join(lines))
    print(f"\nwrote {out_dir}/summary.md, table_main.tex, fig_*.pdf")


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def load_image_meta(cfg) -> dict:
    with open(os.path.join(cfg["cache_dir"], "meta.json")) as f:
        m = json.load(f)
    return {k: v.get("meta", {}) for k, v in m["images"].items() if v.get("aug_of") is None}


def center_bias_cc(cfg) -> dict:
    """Baseline: CC of a central Gaussian map with each image's human fixation map."""
    data = load_all_cache(cfg)
    if data["gaze"] is None:
        return {}
    G = cfg["grid"]["levels"][cfg["grid"]["periph_level"]]
    yy, xx = np.mgrid[0:G, 0:G]
    c = (G - 1) / 2
    cb = np.exp(-(((yy - c) ** 2) + ((xx - c) ** 2)) / (2 * (G / 4) ** 2)).ravel()
    out = {}
    for i, k in enumerate(data["ids"]):
        if data["aug_of"][k] is None and data["has_gaze"][i] and data["gaze"][i].std() > 0:
            out[k] = float(np.corrcoef(cb, data["gaze"][i])[0, 1])
    return out


def make_figures(specs, summary, runs, budget_specs, cfg, out_dir, group_rows=None, group_by=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.titlesize": 9, "axes.titlecolor": INK})

    # Fig 1: forest plot - RL success vs static exhaustive top-1 (left), hacking rate (right)
    y = np.arange(len(specs))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 0.32 * len(specs) + 1.1), sharey=True)
    for k, (key, label) in enumerate([("success", "RL gaze policy"), ("static_top1", "Exhaustive argmax of reward")]):
        vals = [summary[s]["metrics"]["success"] if key == "success" else summary[s]["static_top1"] for s in specs]
        m = np.array([v[0] for v in vals]) * 100
        lo = np.array([v[1] for v in vals]) * 100
        hi = np.array([v[2] for v in vals]) * 100
        off = 0.14 if k == 0 else -0.14
        axes[0].errorbar(m, y + off, xerr=[m - lo, hi - m], fmt=MARKERS[k], ms=5, color=COLORS[k],
                         ecolor=COLORS[k], elinewidth=1.4, capsize=0, label=label, mec="white", mew=0.8)
    hv = [summary[s]["metrics"]["hacked"] for s in specs]
    m = np.array([v[0] for v in hv]) * 100
    axes[1].errorbar(m, y, xerr=[m - np.array([v[1] for v in hv]) * 100, np.array([v[2] for v in hv]) * 100 - m],
                     fmt="o", ms=5, color=COLORS[0], elinewidth=1.4, capsize=0, mec="white", mew=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(specs)
    axes[0].set_xlabel("Held-out success (%)", color=INK)
    axes[1].set_xlabel("Hacked stops (%)", color=INK)
    axes[0].set_title("Finds the target", loc="left")
    axes[1].set_title("Stops on a distractor the reward prefers", loc="left")
    axes[0].legend(frameon=False, loc="lower right", fontsize=7)
    for ax in axes:
        _style(ax)
        ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_success_hacking.pdf"))
    fig.savefig(os.path.join(out_dir, "fig_success_hacking.png"), dpi=200)
    plt.close(fig)

    # Fig 2: success within a glimpse budget (anytime curve), selected specs
    T = cfg["env"]["horizon"]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    xs = np.arange(1, T + 1)
    ends = []
    for k, s in enumerate([b for b in budget_specs if b in summary][:4]):
        per_img = defaultdict(list)
        for r in runs:
            if r["spec"] == s:
                for row in r["test"]:
                    per_img[row["image_id"]].append(row["success_by_step"])
        curve = np.mean([np.mean(v, 0) for v in per_img.values()], 0) * 100
        ax.plot(xs, curve, color=COLORS[k], lw=2, marker=MARKERS[k], ms=3.5, markevery=4, mec="white", mew=0.6, label=s)
        ends.append([curve[-1], s])
    # direct labels at line ends, nudged apart so they never collide
    ends.sort()
    lo, hi = ax.get_ylim()
    gap = 0.08 * (hi - lo)
    for j in range(1, len(ends)):
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + gap)
    ax.set_ylim(lo, max(hi, ends[-1][0] + gap / 2) if ends else hi)
    for y_end, s in ends:
        ax.annotate(s, (xs[-1], y_end), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=INK)
    G0, L = cfg["grid"]["levels"][0], len(cfg["grid"]["levels"])
    ax.set_xlabel("Glimpse budget (steps)", color=INK)
    ax.set_ylabel("Held-out success (%)", color=INK)
    ax.set_title(f"Anytime search (exhaustive scan = {G0 ** 2 * (4 ** (L - 1))} glimpses)", loc="left", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, handlelength=1.5)
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_xlim(1, T + 6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_budget.pdf"))
    fig.savefig(os.path.join(out_dir, "fig_budget.png"), dpi=200)
    plt.close(fig)

    # Fig 3 (optional): dose-response over an image-level factor, e.g. # partial-attribute distractors
    if group_rows:
        show = [b for b in budget_specs if any(g["spec"] == b for g in group_rows)][:4]
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharex=True)
        for k, s_ in enumerate(show):
            rows = sorted([g for g in group_rows if g["spec"] == s_], key=lambda g: g["level"])
            xv = np.arange(len(rows))
            for ax, key in zip(axes, ("success", "hacked")):
                m = np.array([g[key][0] for g in rows]) * 100
                lo_ = np.array([g[key][1] for g in rows]) * 100
                hi_ = np.array([g[key][2] for g in rows]) * 100
                ax.errorbar(xv + (k - 1.5) * 0.06, m, yerr=[m - lo_, hi_ - m], color=COLORS[k], lw=2,
                            marker=MARKERS[k], ms=4, mec="white", mew=0.6, capsize=0, elinewidth=1, label=s_)
        levels_ = sorted({g["level"] for g in group_rows})
        for ax, title in zip(axes, ("Held-out success (%)", "Hacked stops (%)")):
            ax.set_xticks(np.arange(len(levels_)))
            ax.set_xticklabels([str(v) for v in levels_])
            ax.set_xlabel(group_by, color=INK)
            ax.set_title(title, loc="left")
            _style(ax)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], 1.0))
        axes[1].legend(frameon=False, fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_dose_response.pdf"))
        fig.savefig(os.path.join(out_dir, "fig_dose_response.png"), dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
