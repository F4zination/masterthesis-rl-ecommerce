#!/usr/bin/env python3
"""Serve a web dashboard of policy-evaluation metrics.

Reads the artifacts produced by the offline-evaluation pipeline and renders
them as a single self-contained HTML page (no external dependencies), served
over HTTP so it can be opened in a browser:

  * ``sim_eval_all_results.json`` - simulator-oracle eval of the *served*
    policies (PPO=V3, bandit=V2, tabular fallback, baselines). This is the
    apples-to-apples V2-vs-V3 comparison.
  * ``ope_results.json``          - off-policy estimates (IPS/SNIPS/DR/DM + CIs)
    on the held-out split.
  * ``data/train|eval/dataset_summary.json`` - dataset provenance.

Usage::

    # generate + serve (default) at http://127.0.0.1:8050
    python OfflineTraining/policy_dashboard.py

    # just write the static file, do not serve
    python OfflineTraining/policy_dashboard.py --no-serve --out outputs/policy_report.html

The page is rebuilt on every request, so re-running an evaluation and hitting
refresh shows the new numbers immediately.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

# Friendly labels / annotations for the policy keys used by each tool.
_SIM_NOTE = {
    "ppo (V3)": "DemoSiteV3 served policy",
    "bandit (V2)": "DemoSiteV2 served policy (greedy)",
    "tabular (V3 fallback)": "V3 offline fallback",
    "uniform_random": "random baseline",
    "no_op": "never intervene (floor)",
}
_OPE_NOTE = {
    "behavior": "data-collection policy",
    "no_op": "never intervene (floor)",
    "greedy_empirical": "≈ V2 bandit (best empirical arm)",
    "trained_policy": "V3 tabular fallback",
}
_STUDY_POLICIES = {"ppo (V3)", "bandit (V2)"}  # highlight the two A/B arms


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _bar(value: float, vmax: float, color: str) -> str:
    pct = 0.0 if vmax <= 0 else max(0.0, min(100.0, 100.0 * value / vmax))
    return (
        f'<div class="bar"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div>'
        f'<span class="bar-val">{value:.3f}</span></div>'
    )


def _dataset_section() -> str:
    train = _load(_REPO / "OfflineTraining" / "data" / "train" / "dataset_summary.json")
    ev = _load(_REPO / "OfflineTraining" / "data" / "eval" / "dataset_summary.json")
    if not train and not ev:
        return ""
    cells = []
    for label, d in (("Train (policy fitting)", train), ("Eval (held-out OPE)", ev)):
        if not d:
            continue
        cells.append(
            f'<div class="ds"><h4>{label}</h4>'
            f'<p><b>{d.get("n_sessions", "?"):,}</b> sessions &middot; '
            f'<b>{d.get("n_transitions", "?"):,}</b> transitions</p>'
            f'<p class="muted">conv {d.get("conversion_rate", 0):.1%} &middot; '
            f'avg len {d.get("avg_session_length", 0):.2f} &middot; '
            f'mean reward {d.get("reward_mean", 0):.3f}</p></div>'
        )
    return '<section class="card"><h2>Dataset</h2><div class="ds-row">' + "".join(cells) + "</div></section>"


def _sim_section(sim: dict[str, Any] | None) -> str:
    if not sim or not sim.get("policies"):
        return '<section class="card"><h2>Simulator-oracle evaluation</h2><p class="muted">No sim_eval_all_results.json found. Run eval_all_policies_sim.py.</p></section>'
    pol = sim["policies"]
    vmax = max(p["mean_reward_per_session"] for p in pol.values())
    # order: highest reward first
    order = sorted(pol, key=lambda k: pol[k]["mean_reward_per_session"], reverse=True)
    rows = []
    for name in order:
        p = pol[name]
        star = " ★" if name in _STUDY_POLICIES else ""
        cls = ' class="study"' if name in _STUDY_POLICIES else ""
        color = "#2563eb" if name in _STUDY_POLICIES else "#94a3b8"
        rows.append(
            f"<tr{cls}><td><b>{html.escape(name)}</b>{star}<br><span class='muted'>{_SIM_NOTE.get(name, '')}</span></td>"
            f"<td>{_bar(p['mean_reward_per_session'], vmax, color)}</td>"
            f"<td class='num'>{p['std_reward_per_session']:.3f}</td>"
            f"<td class='num'>{p['purchase_rate_approx']:.3f}</td>"
            f"<td class='num'>{p['mean_steps_per_session']:.2f}</td>"
            f"<td class='num'>{p.get('lift_vs_noop_normalized', float('nan')):.2f}</td></tr>"
        )
    verdict = _sim_verdict(pol)
    return (
        f'<section class="card"><h2>Simulator-oracle evaluation '
        f'<span class="sub">({sim.get("n_sessions", "?"):,} sessions/policy, seed {sim.get("seed", "?")})</span></h2>'
        f'{verdict}'
        '<table><thead><tr><th>Policy</th><th>Mean reward / session</th><th>Std</th>'
        '<th>Purchase~</th><th>Steps</th><th>Lift</th></tr></thead><tbody>'
        + "".join(rows)
        + '</tbody></table><p class="muted">Lift: 0 = no-op floor, 1 = uniform-random baseline. '
        '&#9733; = policy actually served in the study.</p></section>'
    )


def _sim_verdict(pol: dict[str, Any]) -> str:
    ppo = pol.get("ppo (V3)", {}).get("mean_reward_per_session")
    ban = pol.get("bandit (V2)", {}).get("mean_reward_per_session")
    if ppo is None or ban is None:
        return ""
    diff = (ppo - ban) / ban * 100 if ban else 0.0
    leader, color = ("V3 (PPO)", "#16a34a") if ppo >= ban else ("V2 (bandit)", "#ea580c")
    return (
        f'<div class="verdict" style="border-color:{color}">'
        f'<b style="color:{color}">{leader} leads</b> &mdash; '
        f'V3 {ppo:.3f} vs V2 {ban:.3f} ({diff:+.1f}% mean reward).</div>'
    )


def _action_section(sim: dict[str, Any] | None) -> str:
    if not sim or not sim.get("policies"):
        return ""
    pol = sim["policies"]
    palette = ["#2563eb", "#16a34a", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#64748b"]
    # stable action ordering across policies
    actions: list[str] = []
    for p in pol.values():
        for a in (p.get("action_distribution") or {}):
            if a not in actions:
                actions.append(a)
    legend = "".join(
        f'<span class="lg"><i style="background:{palette[i % len(palette)]}"></i>{html.escape(a)}</span>'
        for i, a in enumerate(actions)
    )
    rows = []
    for name, p in pol.items():
        dist = p.get("action_distribution") or {}
        total = sum(dist.values()) or 1
        segs = "".join(
            f'<div title="{html.escape(a)}: {dist.get(a,0)}" style="width:{100*dist.get(a,0)/total:.2f}%;'
            f'background:{palette[actions.index(a) % len(palette)]}"></div>'
            for a in actions if dist.get(a, 0)
        )
        rows.append(f'<tr><td>{html.escape(name)}</td><td><div class="stack">{segs}</div></td></tr>')
    return (
        '<section class="card"><h2>Action mix per policy</h2>'
        f'<div class="legend">{legend}</div>'
        '<table class="stacktab"><tbody>' + "".join(rows) + "</tbody></table></section>"
    )


def _ope_section(ope: dict[str, Any] | None) -> str:
    if not ope or not ope.get("policies"):
        return '<section class="card"><h2>Off-policy evaluation (OPE)</h2><p class="muted">No ope_results.json found.</p></section>'
    pol = ope["policies"]
    drs = {k: v["point_estimate"]["dr"] for k, v in pol.items()}
    vmax = max(drs.values())
    order = sorted(pol, key=lambda k: drs[k], reverse=True)
    rows = []
    for name in order:
        pe = pol[name]["point_estimate"]
        ci = pol[name].get("ci95", {}).get("dr", [None, None])
        ci_txt = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci and ci[0] is not None else ""
        color = "#7c3aed" if name in ("greedy_empirical", "trained_policy") else "#94a3b8"
        rows.append(
            f"<tr><td><b>{html.escape(name)}</b><br><span class='muted'>{_OPE_NOTE.get(name, '')}</span></td>"
            f"<td>{_bar(pe['dr'], vmax, color)}<span class='muted'>{ci_txt}</span></td>"
            f"<td class='num'>{pe['ips']:.3f}</td><td class='num'>{pe['snips']:.3f}</td>"
            f"<td class='num'>{pe['dm']:.3f}</td></tr>"
        )
    return (
        f'<section class="card"><h2>Off-policy evaluation '
        f'<span class="sub">(held-out split, {ope.get("n_rows", "?"):,} rows, {ope.get("bootstrap", "?")} bootstraps)</span></h2>'
        '<table><thead><tr><th>Policy</th><th>DR (95% CI)</th><th>IPS</th><th>SNIPS</th><th>DM</th></tr></thead><tbody>'
        + "".join(rows)
        + '</tbody></table><p class="muted">DR = doubly-robust per-decision value (primary OPE estimate). '
        'OPE cannot score the PPO checkpoint &mdash; use the simulator eval above for V3.</p></section>'
    )


_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #f1f5f9; color: #0f172a; }
header { background: #0f172a; color: #fff; padding: 22px 32px; }
header h1 { margin: 0; font-size: 20px; }
header p { margin: 4px 0 0; color: #94a3b8; font-size: 13px; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 60px; }
.card { background: #fff; border-radius: 12px; padding: 20px 24px; margin: 18px 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.card h2 { margin: 0 0 14px; font-size: 16px; }
.card h2 .sub { font-weight: 400; color: #64748b; font-size: 13px; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid #e2e8f0; font-size: 13px; vertical-align: middle; }
th { color: #64748b; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.study { background: #eff6ff; }
.bar { position: relative; background: #f1f5f9; border-radius: 5px; height: 22px; min-width: 160px; }
.bar-fill { height: 100%; border-radius: 5px; }
.bar-val { position: absolute; top: 0; left: 8px; line-height: 22px; font-size: 12px; font-variant-numeric: tabular-nums; }
.muted { color: #94a3b8; font-size: 12px; }
.ds-row { display: flex; gap: 24px; flex-wrap: wrap; }
.ds h4 { margin: 0 0 4px; font-size: 13px; }
.ds p { margin: 2px 0; font-size: 13px; }
.verdict { border-left: 4px solid; background: #f8fafc; padding: 10px 14px; border-radius: 6px; margin-bottom: 14px; font-size: 14px; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 10px; }
.lg { font-size: 12px; color: #475569; display: flex; align-items: center; gap: 5px; }
.lg i { width: 11px; height: 11px; border-radius: 2px; display: inline-block; }
.stack { display: flex; height: 18px; border-radius: 4px; overflow: hidden; background: #f1f5f9; }
.stack div { height: 100%; }
.stacktab td:first-child { width: 180px; }
"""


def build_html() -> str:
    out = _HERE / "outputs"
    sim = _load(out / "sim_eval_all_results.json")
    ope = _load(out / "ope_results.json")
    body = (
        _dataset_section()
        + _sim_section(sim)
        + _action_section(sim)
        + _ope_section(ope)
    )
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Policy Evaluation Dashboard</title>"
        f"<style>{_CSS}</style></head><body>"
        "<header><h1>Policy Evaluation Dashboard</h1>"
        "<p>RL E-Commerce Pipeline &mdash; V2 (bandit) vs V3 (PPO), frozen-policy A/B</p></header>"
        f"<main>{body}</main></body></html>"
    )


def serve(host: str, port: int) -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            payload = build_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):  # silence per-request logging
            pass

    server = HTTPServer((host, port), Handler)
    print(f"Policy dashboard: http://{host}:{port}  (rebuilt on each refresh; Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--no-serve", action="store_true", help="write the HTML file instead of serving")
    ap.add_argument("--out", default=str(_HERE / "outputs" / "policy_report.html"),
                    help="output path when --no-serve is used")
    args = ap.parse_args()

    if args.no_serve:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_html(), encoding="utf-8")
        print(f"Wrote {out}")
    else:
        serve(args.host, args.port)


if __name__ == "__main__":
    main()
