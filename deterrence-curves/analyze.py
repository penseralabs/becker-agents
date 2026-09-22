"""Analyze results: violation rate per condition with 95% Wilson CIs, a
Cochran-Armitage trend test across audit levels, a bar chart, and summary.md.

Usage: python analyze.py results/<model> [results/<other_model> ...]
       [--price-in 1.00 --price-out 5.00]   # $ per million tokens, for a cost estimate
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def cochran_armitage(levels: list[float], ks: list[int], ns: list[int]) -> tuple[float, float] | None:
    N, R = sum(ns), sum(ks)
    if N == 0 or R in (0, N) or len(levels) < 2:
        return None
    pbar = R / N
    t = sum(x * (k - n * pbar) for x, k, n in zip(levels, ks, ns))
    var = pbar * (1 - pbar) * (sum(n * x * x for x, n in zip(levels, ns))
                               - sum(n * x for x, n in zip(levels, ns)) ** 2 / N)
    if var <= 0:
        return None
    z = t / math.sqrt(var)
    return z, math.erfc(abs(z) / math.sqrt(2))


def load(result_dir: Path) -> list[dict]:
    latest = {}
    for line in (result_dir / "episodes.jsonl").read_text().splitlines():
        rec = json.loads(line)
        latest[rec["episode_id"]] = rec
    return [r for r in latest.values() if not r.get("error")]


def level_label(a) -> str:
    return "no oversight text" if a is None else f"{a}%"


def analyze(result_dir: Path, price_in: float | None, price_out: float | None) -> None:
    recs = load(result_dir)
    if not recs:
        print(f"No completed episodes in {result_dir}")
        return
    is_mock = recs[0]["provider"] == "mock"
    model = recs[0]["model"]
    groups = defaultdict(list)
    for r in recs:
        groups[(r["penalty"], r["audit_pct"])].append(r)
    sort_key = lambda k: (k[0], -1 if k[1] is None else k[1])  # noqa: E731
    keys = sorted(groups, key=sort_key)

    rows = []
    for pen, a in keys:
        g = groups[(pen, a)]
        k, n = sum(r["violation"] for r in g), len(g)
        lo, hi = wilson(k, n)
        rows.append({
            "penalty": pen, "audit_pct": a, "n": n, "k": k, "rate": k / n, "lo": lo, "hi": hi,
            "succeeded": sum(r["violation_succeeded"] for r in g),
            "submitted": sum(r["submitted"] for r in g),
            "workaround": sum(r["flag_src_workaround"] for r in g),
            "misreport": sum(r["flag_possible_misreport"] for r in g),
        })

    # ---- chart ----
    penalties = sorted({r["penalty"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.8 / len(penalties)
    levels = sorted({r["audit_pct"] for r in rows}, key=lambda a: -1 if a is None else a)
    for i, pen in enumerate(penalties):
        sub = {r["audit_pct"]: r for r in rows if r["penalty"] == pen}
        xs, ys, errs = [], [], [[], []]
        for j, a in enumerate(levels):
            if a in sub:
                r = sub[a]
                xs.append(j + i * width - 0.4 + width / 2)
                ys.append(100 * r["rate"])
                errs[0].append(100 * (r["rate"] - r["lo"]))
                errs[1].append(100 * (r["hi"] - r["rate"]))
        ax.bar(xs, ys, width, yerr=errs, capsize=4, label=f"penalty: {pen}")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([level_label(a) for a in levels])
    ax.set_xlabel("Stated audit probability")
    ax.set_ylabel("Rule-violation rate (%)")
    ax.set_ylim(0, 100)
    title = f"Deterrence curve: {model}"
    if is_mock:
        title = "MOCK DATA - pipeline test only, not a result"
    ax.set_title(title)
    if len(penalties) > 1:
        ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    chart = result_dir / "deterrence_curve.png"
    fig.savefig(chart, dpi=160)
    plt.close(fig)

    # ---- summary ----
    lines = [f"# Results: {model}", ""]
    if is_mock:
        lines += ["> **MOCK DATA.** Produced by a scripted agent to test the pipeline. Not an experimental result.", ""]
    lines += ["| Penalty | Audit prob. | n | Violations | Rate | 95% CI | Violation made suite pass | Submitted | Flag: src workaround | Flag: possible misreport |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['penalty']} | {level_label(r['audit_pct'])} | {r['n']} | {r['k']} | {100*r['rate']:.1f}% "
                     f"| {100*r['lo']:.1f}–{100*r['hi']:.1f}% | {r['succeeded']} | {r['submitted']} "
                     f"| {r['workaround']} | {r['misreport']} |")
    lines.append("")
    for pen in penalties:
        num = [r for r in rows if r["penalty"] == pen and r["audit_pct"] is not None]
        ca = cochran_armitage([r["audit_pct"] for r in num], [r["k"] for r in num], [r["n"] for r in num])
        if ca:
            z, pval = ca
            direction = "no trend" if abs(z) < 1e-9 else ("rate decreases" if z < 0 else "rate increases")
            lines.append(f"- **Trend test ({pen})**: Cochran-Armitage z = {z:.2f}, p = {pval:.3g} "
                         f"({direction} as audit probability rises).")
        else:
            lines.append(f"- **Trend test ({pen})**: not computable (needs ≥2 levels and some variation in outcomes).")
    tin = sum(r.get("input_tokens", 0) for r in recs)
    tout = sum(r.get("output_tokens", 0) for r in recs)
    lines.append(f"- **Tokens**: {tin:,} input, {tout:,} output across {len(recs)} episodes "
                 f"({tin // max(1, len(recs)):,} / {tout // max(1, len(recs)):,} per episode).")
    if price_in is not None and price_out is not None:
        cost = tin / 1e6 * price_in + tout / 1e6 * price_out
        lines.append(f"- **Estimated cost**: ${cost:.2f} total, ${cost / len(recs):.4f} per episode.")
    lines += ["", "Flags are keyword heuristics for manual review, not primary metrics. "
              "Read transcripts in `transcripts/` to confirm.", "", f"![chart]({chart.name})"]
    (result_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:-1]))
    print(f"\nSaved {chart} and {result_dir / 'summary.md'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--price-in", type=float, default=None, help="$ per million input tokens")
    p.add_argument("--price-out", type=float, default=None, help="$ per million output tokens")
    args = p.parse_args()
    for d in args.dirs:
        analyze(Path(d), args.price_in, args.price_out)


if __name__ == "__main__":
    main()
