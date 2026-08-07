#!/usr/bin/env python3
"""Generate interactive HTML reports for training and evaluation results."""

import json
from pathlib import Path
from typing import Any


def load_policy_stats(policy_path: Path) -> dict[str, Any]:
    """Load statistics from trained policy JSON (handles both FQI/tabular and PPO)."""
    if not policy_path.exists():
        return {}

    with open(policy_path, encoding="utf-8") as f:
        policy = json.load(f)

    algorithm = str(policy.get("algorithm") or "unknown")
    is_ppo = algorithm.lower().startswith("ppo")

    # Action distribution: how many states the greedy policy assigns each action (shared)
    policy_map = policy.get("policy") or {}
    action_counts: dict[str, int] = {}
    for action in policy_map.values():
        action_counts[action] = action_counts.get(action, 0) + 1

    if is_ppo:
        return {
            "is_ppo": True,
            "algorithm": algorithm,
            "gamma": policy.get("gamma", 0),
            "gae_lambda": policy.get("gae_lambda", 0),
            "clip_eps": policy.get("clip_eps", 0),
            "entropy_coef": policy.get("entropy_coef", 0),
            "value_coef": policy.get("value_coef", 0),
            "lr": policy.get("lr", 0),
            "epochs": policy.get("epochs", 0),
            "minibatch_size": policy.get("minibatch_size", 0),
            "n_states": policy.get("n_states", 0),
            "n_policy_states": len(policy_map),
            "n_rows": policy.get("n_rows", 0),
            "state_vocab_size": policy.get("state_vocab_size", 0),
            "action_vocab_size": policy.get("action_vocab_size", 0),
            "timing_mode": policy.get("timing_mode", "unknown"),
            "action_distribution": action_counts,
            "observed_action_counts": {},
            # FQI-only fields kept at neutral values for shared template sections
            "iters": 0,
            "conservative_penalty": 0,
            "q_values_mean": 0.0,
            "q_values_min": 0.0,
            "q_values_max": 0.0,
            "q_values_std": 0.0,
        }

    # ── FQI / tabular policy ────────────────────────────────────────────────────
    q_table = policy.get("q_table") or {}
    q_values = list(q_table.values()) if q_table else []

    observed_action_counts: dict[str, int] = {}
    for key in q_table:
        parts = key.split("|||", 1)
        if len(parts) == 2:
            a = parts[1]
            observed_action_counts[a] = observed_action_counts.get(a, 0) + 1

    return {
        "is_ppo": False,
        "algorithm": algorithm,
        "gamma": policy.get("gamma", 0),
        "gae_lambda": 0,
        "clip_eps": 0,
        "entropy_coef": 0,
        "value_coef": 0,
        "lr": 0,
        "epochs": 0,
        "minibatch_size": 0,
        "iters": policy.get("iters", 0),
        "conservative_penalty": policy.get("conservative_penalty", 0),
        "n_states": policy.get("n_states", 0),
        "n_policy_states": len(policy_map),
        "n_rows": policy.get("n_rows", 0),
        "state_vocab_size": 0,
        "action_vocab_size": 0,
        "timing_mode": policy.get("timing_mode", "unknown"),
        "q_values_mean": sum(q_values) / len(q_values) if q_values else 0,
        "q_values_min": min(q_values) if q_values else 0,
        "q_values_max": max(q_values) if q_values else 0,
        "q_values_std": _std(q_values),
        "action_distribution": action_counts,
        "observed_action_counts": observed_action_counts,
    }


def load_dataset_stats(dataset_path: Path) -> dict[str, Any]:
    """Load statistics from dataset summary JSON."""
    if not dataset_path.exists():
        return {}
    
    with open(dataset_path, encoding="utf-8") as f:
        summary = json.load(f)
    
    return {
        "n_transitions": summary.get("n_transitions", 0),
        "n_sessions": summary.get("n_sessions", 0),
        "reward_mean": summary.get("reward_mean", 0),
        "reward_min": summary.get("reward_min", 0),
        "reward_max": summary.get("reward_max", 0),
        "conversion_rate": summary.get("conversion_rate", 0),
        "avg_session_length": summary.get("avg_session_length", 0),
        "archetype_distribution": summary.get("archetype_distribution", {}),
    }


def load_ope_results(ope_path: Path) -> dict[str, Any]:
    """Load OPE evaluation results."""
    if not ope_path.exists():
        return {}
    
    with open(ope_path, encoding="utf-8") as f:
        return json.load(f)


def load_sim_eval_results(sim_path: Path) -> dict[str, Any]:
    """Load simulator-oracle evaluation results."""
    if not sim_path.exists():
        return {}

    with open(sim_path, encoding="utf-8") as f:
        return json.load(f)


def _std(values: list[float]) -> float:
    """Compute standard deviation."""
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return var ** 0.5


def _fmt_int(value: Any, default: str = "N/A") -> str:
    """Format integer-like values with thousands separators.

    Returns the provided default for missing or non-numeric values so the HTML
    template can safely render partial reports.
    """
    if value is None:
        return default
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return default


def _resolve_existing_path(path_value: str | Path | None, search_roots: list[Path]) -> Path | None:
    """Resolve a possibly-relative artifact path against common workspace roots."""
    if not path_value:
        return None

    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved

    return candidate if candidate.exists() else None


def _resolve_policy_path(out_dir: Path, policy_path: str | None, ope_results: dict[str, Any]) -> Path | None:
    """Resolve the policy artifact path from explicit input or nearby outputs."""
    search_roots = [Path.cwd(), out_dir, out_dir.parent]

    if policy_path:
        return _resolve_existing_path(policy_path, search_roots)

    local_candidate = out_dir / "trained_policy.json"
    if local_candidate.exists():
        return local_candidate

    dataset_value = ope_results.get("dataset")
    if dataset_value:
        dataset_candidate = _resolve_existing_path(dataset_value, search_roots)
        if dataset_candidate:
            sibling_candidate = dataset_candidate.parent / "trained_policy.json"
            if sibling_candidate.exists():
                return sibling_candidate

    return None


def _resolve_dataset_path(out_dir: Path, dataset_path: str | None, ope_results: dict[str, Any]) -> Path | None:
    """Resolve the dataset summary path from explicit input or nearby outputs."""
    search_roots = [Path.cwd(), out_dir, out_dir.parent]

    if dataset_path:
        return _resolve_existing_path(dataset_path, search_roots)

    local_candidate = out_dir / "dataset_summary.json"
    if local_candidate.exists():
        return local_candidate

    dataset_value = ope_results.get("dataset")
    if dataset_value:
        dataset_candidate = _resolve_existing_path(dataset_value, search_roots)
        if dataset_candidate:
            summary_candidate = dataset_candidate.parent / "dataset_summary.json"
            if summary_candidate.exists():
                return summary_candidate

    return None


def _resolve_ope_path(out_dir: Path, ope_path: str | None) -> Path | None:
    """Resolve OPE results path from explicit input or default outputs."""
    search_roots = [Path.cwd(), out_dir, out_dir.parent]

    if ope_path:
        return _resolve_existing_path(ope_path, search_roots)

    local_candidate = out_dir / "ope_results.json"
    if local_candidate.exists():
        return local_candidate

    return None


def _resolve_sim_eval_path(out_dir: Path, sim_path: str | None) -> Path | None:
    """Resolve simulator evaluation path from explicit input or default outputs."""
    search_roots = [Path.cwd(), out_dir, out_dir.parent]

    if sim_path:
        return _resolve_existing_path(sim_path, search_roots)

    local_candidate = out_dir / "sim_eval_results.json"
    if local_candidate.exists():
        return local_candidate

    return None


def _render_training_config_section(policy_stats: dict) -> str:
    """Return the Training Configuration HTML section, branching on algorithm type."""
    algorithm = policy_stats.get("algorithm", "N/A")
    is_ppo = bool(policy_stats.get("is_ppo"))

    if is_ppo:
        return f"""
            <div class="section">
                <h2>\U0001f393 Training Configuration (PPO Actor-Critic)</h2>
                <div class="grid">
                    <div class="card">
                        <h3>Algorithm</h3>
                        <div class="value" style="font-size: 1.3em;">{algorithm}</div>
                    </div>
                    <div class="card">
                        <h3>Discount Factor (\u03b3)</h3>
                        <div class="value">{policy_stats.get("gamma", 0)}</div>
                    </div>
                    <div class="card">
                        <h3>GAE Lambda (\u03bb)</h3>
                        <div class="value">{policy_stats.get("gae_lambda", 0)}</div>
                    </div>
                    <div class="card">
                        <h3>Clip \u03b5</h3>
                        <div class="value">{policy_stats.get("clip_eps", 0)}</div>
                    </div>
                    <div class="card">
                        <h3>Entropy Coef</h3>
                        <div class="value">{policy_stats.get("entropy_coef", 0)}</div>
                    </div>
                    <div class="card">
                        <h3>Value Coef</h3>
                        <div class="value">{policy_stats.get("value_coef", 0)}</div>
                    </div>
                    <div class="card">
                        <h3>Learning Rate</h3>
                        <div class="value">{policy_stats.get("lr", 0)}</div>
                    </div>
                    <div class="card">
                        <h3>PPO Epochs</h3>
                        <div class="value">{policy_stats.get("epochs", 0)}</div>
                    </div>
                    <div class="card">
                        <h3>Minibatch Size</h3>
                        <div class="value">{policy_stats.get("minibatch_size", 0)}</div>
                    </div>
                </div>
            </div>"""
    else:
        return f"""
            <div class="section">
                <h2>\U0001f393 Training Configuration</h2>
                <div class="grid">
                    <div class="card">
                        <h3>Algorithm</h3>
                        <div class="value" style="font-size: 1.3em;">{algorithm}</div>
                    </div>
                    <div class="card">
                        <h3>Discount Factor (\u03b3)</h3>
                        <div class="value">{policy_stats.get("gamma", 0)}</div>
                    </div>
                    <div class="card">
                        <h3>FQI Iterations</h3>
                        <div class="value">{policy_stats.get("iters", 0)}</div>
                    </div>
                    <div class="card">
                        <h3>Conservative Penalty</h3>
                        <div class="value">{policy_stats.get("conservative_penalty", 0)}</div>
                    </div>
                </div>
            </div>"""


def _render_policy_chart_section(policy_stats: dict) -> str:
    """Return the top-of-policy-stats chart block: vocab cards for PPO, Q-value doughnut for FQI."""
    if bool(policy_stats.get("is_ppo")):
        vocab_size = policy_stats.get("state_vocab_size", "N/A")
        action_vocab = policy_stats.get("action_vocab_size", "N/A")
        n_rows = int(policy_stats.get("n_rows") or 0)
        return f"""
                <div class="grid" style="margin-bottom:24px;">
                    <div class="card">
                        <h3>State Vocab Size</h3>
                        <div class="value">{vocab_size}</div>
                        <div class="unit">one-hot input dimensions</div>
                    </div>
                    <div class="card">
                        <h3>Action Vocab Size</h3>
                        <div class="value">{action_vocab}</div>
                        <div class="unit">output logits (softmax)</div>
                    </div>
                    <div class="card">
                        <h3>Training Samples</h3>
                        <div class="value">{n_rows:,}</div>
                        <div class="unit">JSONL transitions used</div>
                    </div>
                </div>"""
    else:
        return """
                <div class="row full">
                    <div class="chart-container">
                        <div class="chart-title">Q-Value Distribution</div>
                        <canvas id="qvalueChart"></canvas>
                    </div>
                </div>"""


def _render_policy_extra_stats(policy_stats: dict) -> str:
    """Return the extra stats rows below the action-distribution charts."""
    if bool(policy_stats.get("is_ppo")):
        vocab_size = policy_stats.get("state_vocab_size", 0)
        action_vocab = policy_stats.get("action_vocab_size", 0)
        n_rows = int(policy_stats.get("n_rows") or 0)
        return f"""
                <div class="row full">
                    <div>
                        <p style="color: #666; margin-bottom: 12px;">
                            <strong>Metric note:</strong> PPO uses a neural actor-critic; no Q-table is maintained.
                            State Coverage = greedy policy states / observed states.
                            Vocab sizes reflect the bag-of-tokens feature dimensions used at inference time.
                        </p>
                        <div class="stat-row">
                            <span class="label">State Vocab Size</span>
                            <span class="value">{vocab_size}</span>
                        </div>
                        <div class="stat-row">
                            <span class="label">Action Vocab Size</span>
                            <span class="value">{action_vocab}</span>
                        </div>
                        <div class="stat-row">
                            <span class="label">Training Samples (n_rows)</span>
                            <span class="value">{n_rows:,}</span>
                        </div>
                    </div>
                </div>"""
    else:
        q_mean = policy_stats.get("q_values_mean", 0)
        q_min = policy_stats.get("q_values_min", 0)
        q_max = policy_stats.get("q_values_max", 0)
        q_std = policy_stats.get("q_values_std", 0)
        return f"""
                <div class="row full">
                    <div>
                        <p style="color: #666; margin-bottom: 12px;">
                            <strong>Metric note:</strong> State Coverage = policy states / observed states.
                            Q-value mean, range, and std summarize the scale and spread of learned action values.
                        </p>
                        <div class="stat-row">
                            <span class="label">Q-Value Mean</span>
                            <span class="value">{q_mean:.3f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="label">Q-Value Range</span>
                            <span class="value">[{q_min:.3f}, {q_max:.3f}]</span>
                        </div>
                        <div class="stat-row">
                            <span class="label">Q-Value Std Dev</span>
                            <span class="value">{q_std:.3f}</span>
                        </div>
                    </div>
                </div>"""


def _render_key_insights(policy_stats: dict, dataset_stats: dict) -> str:
    """Return the Key Insights highlight block, adapted to algorithm type."""
    n_transitions = int(dataset_stats.get("n_transitions") or 0)
    n_sessions = int(dataset_stats.get("n_sessions") or 0)
    n_policy = int(policy_stats.get("n_policy_states") or 0)
    n_states = max(int(policy_stats.get("n_states") or 1), 1)
    coverage = 100.0 * n_policy / n_states

    if bool(policy_stats.get("is_ppo")):
        vocab_size = policy_stats.get("state_vocab_size", "N/A")
        action_vocab = policy_stats.get("action_vocab_size", "N/A")
        clip_eps = policy_stats.get("clip_eps", 0)
        gae_lam = policy_stats.get("gae_lambda", 0)
        return f"""
            <div class="highlight">
                <strong>\U0001f4cc Key Insights (PPO):</strong>
                <ul style="margin: 10px 0 0 20px;">
                    <li>Dataset contains {n_transitions:,} transitions from {n_sessions:,} sessions</li>
                    <li>PPO greedy policy covers {n_policy:,} unique states ({coverage:.1f}% of observed states)</li>
                    <li>Network: input dim {vocab_size} (state vocab) \u00b7 output dim {action_vocab} (action vocab)</li>
                    <li>Clip \u03b5 = {clip_eps} \u00b7 GAE \u03bb = {gae_lam}</li>
                </ul>
            </div>"""
    else:
        q_mean = policy_stats.get("q_values_mean", 0)
        q_min = policy_stats.get("q_values_min", 0)
        q_max = policy_stats.get("q_values_max", 0)
        return f"""
            <div class="highlight">
                <strong>\U0001f4cc Key Insights:</strong>
                <ul style="margin: 10px 0 0 20px;">
                    <li>Dataset contains {n_transitions:,} transitions from {n_sessions:,} sessions</li>
                    <li>Policy learned Q-values for {n_policy:,} unique states ({coverage:.1f}% coverage)</li>
                    <li>Q-value statistics: mean {q_mean:.2f}, range [{q_min:.2f}, {q_max:.2f}]</li>
                </ul>
            </div>"""


def _render_qvalue_chart_js(policy_stats: dict) -> str:
    """Return the Q-value Distribution JS chart block (empty for PPO policies)."""
    if bool(policy_stats.get("is_ppo")):
        return "        // Q-Value Distribution chart not applicable for PPO policies."
    q_max = policy_stats.get("q_values_max", 0)
    q_min = policy_stats.get("q_values_min", 0)
    q_mean = policy_stats.get("q_values_mean", 0)
    q_std = policy_stats.get("q_values_std", 0)
    return f"""        // Q-Value Distribution
        const qvalueCtx = document.getElementById('qvalueChart');
        if (qvalueCtx) {{
            new Chart(qvalueCtx, {{
                type: 'doughnut',
                data: {{
                    labels: ['Q-Value Range', 'Mean', 'Std Dev'],
                    datasets: [{{
                        data: [
                            Math.abs({q_max} - {q_min}),
                            {q_mean},
                            {q_std}
                        ],
                        backgroundColor: ['#667eea', '#764ba2', '#f093fb'],
                        borderColor: 'white',
                        borderWidth: 2,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'right',
                        }}
                    }}
                }}
            }});
        }}"""


def generate_html_report(
    output_dir: str,
    policy_path: str | None = None,
    dataset_path: str | None = None,
    ope_path: str | None = None,
    sim_path: str | None = None,
) -> Path:
    """Generate comprehensive HTML report with training and evaluation visualizations.
    
    Args:
        output_dir: Directory to save HTML report
        policy_path: Path to trained_policy.json (optional)
        dataset_path: Path to dataset_summary.json (optional)
        ope_path: Path to ope_results.json (optional)
        sim_path: Path to sim_eval_results.json (optional)
    
    Returns:
        Path to generated HTML file
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_ope_path = _resolve_ope_path(out_dir, ope_path)
    resolved_sim_path = _resolve_sim_eval_path(out_dir, sim_path)

    ope_results = load_ope_results(resolved_ope_path) if resolved_ope_path else {}
    sim_eval_results = load_sim_eval_results(resolved_sim_path) if resolved_sim_path else {}
    resolved_policy_path = _resolve_policy_path(out_dir, policy_path, ope_results)
    resolved_dataset_path = _resolve_dataset_path(out_dir, dataset_path, ope_results)

    policy_stats = load_policy_stats(resolved_policy_path) if resolved_policy_path else {}
    dataset_stats = load_dataset_stats(resolved_dataset_path) if resolved_dataset_path else {}

    sources = {
        "report": str((out_dir / "report.html").resolve()),
        "policy": str(resolved_policy_path) if resolved_policy_path else None,
        "dataset": str(resolved_dataset_path) if resolved_dataset_path else None,
        "ope": str(resolved_ope_path) if resolved_ope_path and resolved_ope_path.exists() else None,
        "sim": str(resolved_sim_path) if resolved_sim_path and resolved_sim_path.exists() else None,
    }

    html = _build_html(policy_stats, dataset_stats, ope_results, sim_eval_results, sources)
    
    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    
    return report_path


def _build_html(
    policy_stats: dict,
    dataset_stats: dict,
    ope_results: dict,
    sim_eval_results: dict,
    sources: dict[str, str | None],
) -> str:
    """Build HTML report with embedded JavaScript visualizations."""
    
    archetype_data = _format_archetype_chart(dataset_stats.get("archetype_distribution", {}))
    ope_comparison = _format_ope_chart(ope_results)
    sim_comparison = _format_sim_chart(sim_eval_results)
    training_config_section = _render_training_config_section(policy_stats)
    policy_chart_section = _render_policy_chart_section(policy_stats)
    policy_extra_stats = _render_policy_extra_stats(policy_stats)
    key_insights_section = _render_key_insights(policy_stats, dataset_stats)
    qvalue_chart_js = _render_qvalue_chart_js(policy_stats)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Offline RL Training & Evaluation Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
            overflow: hidden;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        header p {{
            font-size: 1.1em;
            opacity: 0.95;
            font-weight: 300;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 50px;
        }}
        .section h2 {{
            font-size: 1.8em;
            color: #333;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
            display: inline-block;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }}
        .card {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 25px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
        }}
        .card h3 {{
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        .card .value {{
            font-size: 2.2em;
            color: #333;
            font-weight: 700;
            margin-bottom: 5px;
        }}
        .card .unit {{
            color: #999;
            font-size: 0.85em;
            font-weight: 400;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 30px 0;
            background: white;
            border-radius: 8px;
            border: 1px solid #e9ecef;
            padding: 20px;
        }}
        .chart-title {{
            font-size: 1.3em;
            color: #333;
            margin-bottom: 20px;
            font-weight: 600;
        }}
        .row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        .row.full {{
            grid-template-columns: 1fr;
        }}
        footer {{
            background: #f8f9fa;
            border-top: 1px solid #e9ecef;
            padding: 30px 40px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        .highlight {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
            color: #856404;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin: 12px 0;
            padding: 8px 0;
            border-bottom: 1px solid #e9ecef;
        }}
        .stat-row .label {{
            color: #666;
            font-weight: 500;
        }}
        .stat-row .value {{
            color: #333;
            font-weight: 600;
        }}
        .badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin: 5px 5px 5px 0;
        }}
        .badge.success {{ background: #28a745; }}
        .badge.warning {{ background: #ffc107; color: #333; }}
        .badge.danger {{ background: #dc3545; }}
        .artifact-list {{
            display: grid;
            gap: 12px;
            margin: 20px 0 30px;
        }}
        .artifact-item {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 14px 16px;
        }}
        .artifact-label {{
            color: #667eea;
            font-size: 0.85em;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }}
        .artifact-path {{
            color: #333;
            font-family: Consolas, "SFMono-Regular", Menlo, monospace;
            font-size: 0.9em;
            line-height: 1.4;
            word-break: break-all;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 Offline RL Training & Evaluation</h1>
            <p>Comprehensive Report: Policy Training Results & OPE Analysis</p>
        </header>
        
        <div class="content">
            <div class="section">
                <h2>📁 Loaded Artifacts</h2>
                {_render_artifact_sources(sources)}
            </div>

            <!-- Dataset Overview -->
            <div class="section">
                <h2>📊 Dataset Overview</h2>
                <div class="grid">
                    <div class="card">
                        <h3>Total Transitions</h3>
                        <div class="value">{_fmt_int(dataset_stats.get("n_transitions"))}</div>
                        <div class="unit">training samples</div>
                    </div>
                    <div class="card">
                        <h3>Sessions</h3>
                        <div class="value">{_fmt_int(dataset_stats.get("n_sessions"))}</div>
                        <div class="unit">customer journeys</div>
                    </div>
                    <div class="card">
                        <h3>Avg Session Length</h3>
                        <div class="value">{dataset_stats.get("avg_session_length", 0):.1f}</div>
                        <div class="unit">transitions/session</div>
                    </div>
                    <div class="card">
                        <h3>Conversion Rate</h3>
                        <div class="value">{dataset_stats.get("conversion_rate", 0):.1%}</div>
                        <div class="unit">sessions with purchase</div>
                    </div>
                </div>

                <p style="color: #666; margin: -10px 0 20px;">
                    <strong>Metric note:</strong> Conversion Rate = sessions with at least one purchase / total sessions.
                    Avg Session Length is the average number of transitions per session.
                </p>
                
                {_render_archetype_section(archetype_data)}
                
                <div class="chart-container">
                    <div class="chart-title">📈 Reward Distribution</div>
                    <canvas id="rewardHistogram"></canvas>
                </div>
            </div>
            
            {training_config_section}
            
            <!-- Policy Statistics -->
            <div class="section">
                <h2>🎯 Policy Statistics</h2>
                {policy_chart_section}
                <div class="grid">
                    <div class="card">
                        <h3>Unique States</h3>
                        <div class="value">{_fmt_int(policy_stats.get("n_states"))}</div>
                        <div class="unit">state-action pairs seen</div>
                    </div>
                    <div class="card">
                        <h3>Policy States</h3>
                        <div class="value">{_fmt_int(policy_stats.get("n_policy_states"))}</div>
                        <div class="unit">states with learned actions</div>
                    </div>
                    <div class="card">
                        <h3>State Coverage</h3>
                        <div class="value">{100 * policy_stats.get("n_policy_states", 0) / max(policy_stats.get("n_states", 1), 1):.1f}%</div>
                        <div class="unit">of observed states</div>
                    </div>
                </div>
                <div class="row full">
                    <div class="chart-container" style="height:360px;">
                        <div class="chart-title">🎬 Greedy Action Distribution (States per Action)</div>
                        <canvas id="actionDistChart"></canvas>
                    </div>
                </div>
                <div class="row full">
                    <div class="chart-container" style="height:360px;">
                        <div class="chart-title">🔍 Observed State–Action Pairs per Action</div>
                        <canvas id="observedActionChart"></canvas>
                    </div>
                </div>
                <p style="color: #666; margin: -10px 0 20px;">
                    <strong>Metric note:</strong>
                    <em>Greedy Action Distribution</em> counts how many states the learned policy maps to each action (one action per state).
                    <em>Observed Pairs</em> counts distinct (state, action) entries in the Q-table — reflecting how much training data each action received.
                </p>
                {policy_extra_stats}
            </div>
            
            {_render_ope_section(ope_comparison)}
            {_render_sim_section(sim_comparison)}
            
            {key_insights_section}
        </div>
        
        <footer>
            <p>Generated by OfflineTraining pipeline • Offline Reinforcement Learning for E-Commerce • {_get_timestamp()}</p>
        </footer>
    </div>
    
    <script>
        // Reward Histogram
        const rewardCtx = document.getElementById('rewardHistogram');
        if (rewardCtx) {{
            new Chart(rewardCtx, {{
                type: 'bar',
                data: {{
                    labels: ['Min', 'Q1', 'Median', 'Q3', 'Max'],
                    datasets: [{{
                        label: 'Reward',
                        data: [{dataset_stats.get("reward_min", 0)}, {dataset_stats.get("reward_min", 0) + (dataset_stats.get("reward_max", 0) - dataset_stats.get("reward_min", 0)) * 0.25}, {dataset_stats.get("reward_mean", 0)}, {dataset_stats.get("reward_min", 0) + (dataset_stats.get("reward_max", 0) - dataset_stats.get("reward_min", 0)) * 0.75}, {dataset_stats.get("reward_max", 0)}],
                        backgroundColor: ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1', '#5f27cd'],
                        borderRadius: 6,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        y: {{ beginAtZero: true }}
                    }}
                }}
            }});
        }}
        
        {qvalue_chart_js}
        
        {_render_ope_charts(ope_comparison)}
        {_render_sim_charts(sim_comparison)}
        {_render_action_distribution_charts(policy_stats)}
    </script>
</body>
</html>"""
    
    return html


def _render_archetype_section(archetype_data: dict) -> str:
    """Render archetype distribution section."""
    if not archetype_data.get("names"):
        return ""
    
    return f"""
                <div class="chart-container">
                    <div class="chart-title">👥 Archetype Distribution</div>
                    <canvas id="archetypeChart"></canvas>
                </div>
                <script>
                    const arcCtx = document.getElementById('archetypeChart');
                    if (arcCtx) {{
                        new Chart(arcCtx, {{
                            type: 'pie',
                            data: {{
                                labels: {json.dumps(archetype_data["names"])},
                                datasets: [{{
                                    data: {json.dumps(archetype_data["counts"])},
                                    backgroundColor: ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b', '#fa709a', '#fee140', '#30b0fe', '#ec008c'],
                                    borderColor: 'white',
                                    borderWidth: 2,
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {{
                                    legend: {{
                                        position: 'bottom',
                                    }}
                                }}
                            }}
                        }});
                    }}
                </script>
"""


def _render_artifact_sources(sources: dict[str, str | None]) -> str:
    """Render the resolved artifact source paths used for this report."""
    labels = {
        "report": "Report Output",
        "policy": "Policy Artifact",
        "dataset": "Dataset Summary",
        "ope": "OPE Results",
        "sim": "Simulator Eval Results",
    }
    items = []
    for key in ["report", "policy", "dataset", "ope", "sim"]:
        path_value = sources.get(key)
        items.append(
            f"""
                <div class=\"artifact-item\">
                    <div class=\"artifact-label\">{labels[key]}</div>
                    <div class=\"artifact-path\">{path_value or 'Not loaded'}</div>
                </div>
            """
        )
    return f"<div class=\"artifact-list\">{''.join(items)}</div>"


def _render_ope_section(ope_comparison: dict) -> str:
    """Render OPE evaluation section."""
    if not ope_comparison:
        return ""

    split_badge = ""
    if ope_comparison.get("has_split_eval"):
        split_badge = (
            '<span class="badge success" style="margin-left:12px;">Train/Eval Split</span>'
            f' <span style="color:#666;font-size:0.85em;">train: {ope_comparison["train_dataset"]}</span>'
        )

    temp_val = ope_comparison.get("temperature")
    temp_note = ""
    if temp_val is not None and float(temp_val) > 0:
        temp_note = f'<li>Softmax temperature <strong>{temp_val}</strong> applied to trained policy (non-zero IPS weights for all logged actions).</li>'

    lift_display = ope_comparison.get("lift_display", "N/A")

    return f"""
            <!-- Offline Policy Evaluation -->
            <div class="section">
                <h2>🔬 Offline Policy Evaluation (OPE) {split_badge}</h2>
                <p style="color: #666; margin-bottom: 20px;">
                    Estimates of policy performance using IPS, SNIPS, DR, and DM estimators.
                    Confidence intervals computed via bootstrap resampling.
                </p>

                <div class="highlight">
                    <strong>Estimator Guide:</strong>
                    <ul style="margin: 10px 0 0 20px;">
                        <li><strong>IPS</strong> (Inverse Propensity Scoring): unbiased under correct propensities, but can be high-variance.</li>
                        <li><strong>SNIPS</strong> (Self-Normalized IPS): normalizes IPS weights for more stable finite-sample behavior.</li>
                        <li><strong>DR</strong> (Doubly Robust): combines a reward model with importance weighting and is often most stable.</li>
                        <li><strong>DM</strong> (Direct Method): reward-model-only estimate; no importance weights — lower variance but higher bias. Useful as a sanity check alongside DR.</li>
                        <li><strong>95% CI</strong>: bootstrap confidence interval over resampled datasets; narrower intervals indicate lower estimator variance.</li>
                        {temp_note}
                    </ul>
                </div>

                <div class="row full">
                    <div class="chart-container">
                        <div class="chart-title">Policy Performance Comparison (DR)</div>
                        <canvas id="opeComparisonChart"></canvas>
                    </div>
                </div>

                <div class="grid">
                    <div class="card">
                        <h3>Best Policy (DR)</h3>
                        <div class="value">{ope_comparison.get("best_mean", "N/A")}</div>
                        <div class="unit">expected reward</div>
                    </div>
                    <div class="card">
                        <h3>Improvement vs Behavior</h3>
                        <div class="value">{ope_comparison.get("improvement", "N/A")}</div>
                        <div class="unit">absolute DR delta</div>
                    </div>
                    <div class="card">
                        <h3>Normalized Lift (DR)</h3>
                        <div class="value">{lift_display}</div>
                        <div class="unit">(trained−noop) / (behavior−noop) · >1 beats behavior</div>
                    </div>
                </div>

                <div class="row full">
                    <div class="chart-container">
                        <div class="chart-title">Estimator Comparison — IPS · SNIPS · DR · DM (Point Estimates)</div>
                        <canvas id="estimatorChart"></canvas>
                    </div>
                </div>
            </div>
"""


def _render_ope_charts(ope_comparison: dict) -> str:
    """Render OPE evaluation JavaScript charts."""
    if not ope_comparison:
        return ""

    comparison_data = ope_comparison.get("comparison_data", {})
    policy_names = ope_comparison.get("policy_names", [])
    estimator_data = ope_comparison.get("estimator_data", {})

    return f"""
        // OPE Comparison
        const opeCtx = document.getElementById('opeComparisonChart');
        if (opeCtx) {{
            new Chart(opeCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(list(comparison_data.keys()))},
                    datasets: [
                        {{
                            label: 'DR Estimate',
                            data: {json.dumps([comparison_data.get(k, {}).get("dr_mean", 0) for k in comparison_data.keys()])},
                            backgroundColor: '#667eea',
                            borderRadius: 6,
                        }},
                        {{
                            label: 'DM Estimate',
                            data: {json.dumps([comparison_data.get(k, {}).get("dm_mean", 0) for k in comparison_data.keys()])},
                            backgroundColor: '#f093fb',
                            borderRadius: 6,
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ position: 'top' }}
                    }},
                    scales: {{
                        y: {{ beginAtZero: true }}
                    }}
                }}
            }});
        }}

        // Estimator Comparison
        const estCtx = document.getElementById('estimatorChart');
        if (estCtx) {{
            new Chart(estCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(policy_names)},
                    datasets: [
                        {{
                            label: 'IPS',
                            data: {json.dumps(estimator_data.get("ips", []))},
                            backgroundColor: '#ff6b6b',
                            borderRadius: 6,
                        }},
                        {{
                            label: 'SNIPS',
                            data: {json.dumps(estimator_data.get("snips", []))},
                            backgroundColor: '#feca57',
                            borderRadius: 6,
                        }},
                        {{
                            label: 'DR',
                            data: {json.dumps(estimator_data.get("dr", []))},
                            backgroundColor: '#48dbfb',
                            borderRadius: 6,
                        }},
                        {{
                            label: 'DM',
                            data: {json.dumps(estimator_data.get("dm", []))},
                            backgroundColor: '#1dd1a1',
                            borderRadius: 6,
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'top',
                        }}
                    }},
                    scales: {{
                        y: {{ beginAtZero: true }}
                    }}
                }}
            }});
        }}
"""


def _render_sim_section(sim_comparison: dict) -> str:
    """Render simulator evaluation section."""
    if not sim_comparison:
        return ""

    return f"""
            <!-- Simulator Oracle Evaluation -->
            <div class="section">
                <h2>🧪 Simulator Oracle Evaluation</h2>
                <p style="color: #666; margin-bottom: 20px;">
                    Head-to-head on-policy rollout in CustomerSimulation for trained, uniform-random,
                    and no-op policies.
                </p>

                <div class="grid">
                    <div class="card">
                        <h3>Best Policy (Mean Reward)</h3>
                        <div class="value">{sim_comparison.get("best_policy", "N/A")}</div>
                        <div class="unit">simulator ground truth</div>
                    </div>
                    <div class="card">
                        <h3>Trained vs Uniform Δ</h3>
                        <div class="value">{sim_comparison.get("delta_vs_uniform", "N/A")}</div>
                        <div class="unit">mean reward/session</div>
                    </div>
                    <div class="card">
                        <h3>Normalized Lift (Sim)</h3>
                        <div class="value">{sim_comparison.get("lift_display", "N/A")}</div>
                        <div class="unit">(trained−noop) / (uniform−noop)</div>
                    </div>
                    <div class="card">
                        <h3>Sessions per Policy</h3>
                        <div class="value">{_fmt_int(sim_comparison.get("n_sessions"))}</div>
                        <div class="unit">rollouts</div>
                    </div>
                </div>

                <div class="row full">
                    <div class="chart-container">
                        <div class="chart-title">Mean Reward per Session (Simulator)</div>
                        <canvas id="simRewardChart"></canvas>
                    </div>
                </div>

                <div class="row full">
                    <div class="chart-container" style="height:360px;">
                        <div class="chart-title">Purchase Rate Approximation (Simulator)</div>
                        <canvas id="simPurchaseChart"></canvas>
                    </div>
                </div>
            </div>
"""


def _render_sim_charts(sim_comparison: dict) -> str:
    """Render simulator evaluation JavaScript charts."""
    if not sim_comparison:
        return ""

    policy_names = sim_comparison.get("policy_names", [])
    reward_values = sim_comparison.get("reward_values", [])
    purchase_values = sim_comparison.get("purchase_values", [])

    return f"""
        // Simulator reward comparison
        const simRewardCtx = document.getElementById('simRewardChart');
        if (simRewardCtx) {{
            new Chart(simRewardCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(policy_names)},
                    datasets: [{{
                        label: 'Mean reward/session',
                        data: {json.dumps(reward_values)},
                        backgroundColor: ['#1dd1a1', '#667eea', '#ff6b6b'],
                        borderRadius: 6,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{ y: {{ beginAtZero: true }} }}
                }}
            }});
        }}

        // Simulator purchase-rate comparison
        const simPurchaseCtx = document.getElementById('simPurchaseChart');
        if (simPurchaseCtx) {{
            new Chart(simPurchaseCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(policy_names)},
                    datasets: [{{
                        label: 'Purchase rate (approx)',
                        data: {json.dumps(purchase_values)},
                        backgroundColor: ['#43e97b', '#48dbfb', '#feca57'],
                        borderRadius: 6,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 1,
                            ticks: {{ callback: (v) => (v * 100).toFixed(0) + '%' }}
                        }}
                    }}
                }}
            }});
        }}
"""


def _render_action_distribution_charts(policy_stats: dict) -> str:
    """Render JS for action distribution charts."""
    action_dist = policy_stats.get("action_distribution", {})
    obs_counts = policy_stats.get("observed_action_counts", {})

    if not action_dist and not obs_counts:
        return ""

    all_labels = sorted(set(action_dist) | set(obs_counts))
    greedy_counts = [action_dist.get(k, 0) for k in all_labels]
    observed_counts = [obs_counts.get(k, 0) for k in all_labels]

    return f"""
        // Greedy Action Distribution
        const actionDistCtx = document.getElementById('actionDistChart');
        if (actionDistCtx) {{
            new Chart(actionDistCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(all_labels)},
                    datasets: [{{
                        label: 'States',
                        data: {json.dumps(greedy_counts)},
                        backgroundColor: '#667eea',
                        borderRadius: 6,
                    }}]
                }},
                options: {{
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        x: {{ beginAtZero: true, title: {{ display: true, text: 'Number of States' }} }}
                    }}
                }}
            }});
        }}

        // Observed State-Action Pairs per Action
        const obsActionCtx = document.getElementById('observedActionChart');
        if (obsActionCtx) {{
            new Chart(obsActionCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(all_labels)},
                    datasets: [{{
                        label: 'Observed (state, action) pairs',
                        data: {json.dumps(observed_counts)},
                        backgroundColor: '#764ba2',
                        borderRadius: 6,
                    }}]
                }},
                options: {{
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        x: {{ beginAtZero: true, title: {{ display: true, text: 'Number of (state, action) pairs' }} }}
                    }}
                }}
            }});
        }}
"""


def _format_archetype_chart(archetype_dist: dict) -> dict:
    """Format archetype distribution for charting."""
    if not archetype_dist:
        return {}
    return {
        "names": list(archetype_dist.keys()),
        "counts": list(archetype_dist.values()),
    }


def _format_ope_chart(ope_results: dict) -> dict:
    """Format OPE results for charting."""
    if not ope_results:
        return {}

    policies = ope_results.get("policies", {})
    if not isinstance(policies, dict) or not policies:
        return {}

    comparison_data = {}
    policy_names = []
    estimator_data = {"ips": [], "snips": [], "dr": [], "dm": []}
    best_mean = float("-inf")
    best_policy = ""

    for policy_name, estimates in policies.items():
        if not isinstance(estimates, dict):
            continue

        point_estimate = estimates.get("point_estimate", {})
        ci95 = estimates.get("ci95", {})
        if not isinstance(point_estimate, dict):
            continue

        dr_mean = float(point_estimate.get("dr", 0) or 0)
        dr_ci = ci95.get("dr", [dr_mean, dr_mean]) if isinstance(ci95, dict) else [dr_mean, dr_mean]
        if not isinstance(dr_ci, list) or len(dr_ci) != 2:
            dr_ci = [dr_mean, dr_mean]

        dm_mean = float(point_estimate.get("dm", 0) or 0)
        dm_ci = ci95.get("dm", [dm_mean, dm_mean]) if isinstance(ci95, dict) else [dm_mean, dm_mean]
        if not isinstance(dm_ci, list) or len(dm_ci) != 2:
            dm_ci = [dm_mean, dm_mean]

        comparison_data[policy_name] = {
            "dr_mean": dr_mean,
            "ci_lower": float(dr_ci[0] or dr_mean),
            "ci_upper": float(dr_ci[1] or dr_mean),
            "dm_mean": dm_mean,
        }

        policy_names.append(policy_name)
        estimator_data["ips"].append(float(point_estimate.get("ips", 0) or 0))
        estimator_data["snips"].append(float(point_estimate.get("snips", 0) or 0))
        estimator_data["dr"].append(dr_mean)
        estimator_data["dm"].append(dm_mean)

        if dr_mean > best_mean:
            best_mean = dr_mean
            best_policy = policy_name

    behavior_dr = comparison_data.get("behavior", {}).get("dr_mean", 0)
    improvement = best_mean - behavior_dr if best_mean != float("-inf") else 0
    display_best_mean = f"{best_policy}: {best_mean:.3f}" if best_policy else "N/A"

    raw_lift = ope_results.get("lift_vs_noop_normalized")
    lift_display = f"{raw_lift:.3f}" if isinstance(raw_lift, (int, float)) and not (raw_lift != raw_lift) else "N/A"

    train_dataset = ope_results.get("train_dataset")
    temperature = ope_results.get("temperature")

    return {
        "comparison_data": comparison_data,
        "policy_names": policy_names,
        "estimator_data": estimator_data,
        "best_mean": display_best_mean,
        "improvement": f"{improvement:.3f}",
        "lift_display": lift_display,
        "has_split_eval": bool(train_dataset),
        "train_dataset": str(train_dataset) if train_dataset else None,
        "temperature": temperature,
    }


def _format_sim_chart(sim_results: dict) -> dict:
    """Format simulator results for rendering."""
    if not sim_results:
        return {}

    policies = sim_results.get("policies", {})
    if not isinstance(policies, dict) or not policies:
        return {}

    ordered_names = [
        name for name in ["trained_policy", "uniform_random", "no_op"] if name in policies
    ]
    for name in policies:
        if name not in ordered_names:
            ordered_names.append(name)

    reward_values = [
        float((policies.get(name, {}) or {}).get("mean_reward_per_session", 0) or 0)
        for name in ordered_names
    ]
    purchase_values = [
        float((policies.get(name, {}) or {}).get("purchase_rate_approx", 0) or 0)
        for name in ordered_names
    ]

    best_idx = max(range(len(ordered_names)), key=lambda i: reward_values[i]) if ordered_names else -1
    best_policy = (
        f"{ordered_names[best_idx]}: {reward_values[best_idx]:.3f}"
        if best_idx >= 0
        else "N/A"
    )

    trained = float((policies.get("trained_policy", {}) or {}).get("mean_reward_per_session", 0) or 0)
    uniform = float((policies.get("uniform_random", {}) or {}).get("mean_reward_per_session", 0) or 0)
    delta_vs_uniform = trained - uniform

    raw_lift = sim_results.get("lift_vs_noop_normalized")
    lift_display = (
        f"{raw_lift:.3f}"
        if isinstance(raw_lift, (int, float)) and not (raw_lift != raw_lift)
        else "N/A"
    )

    n_sessions = (policies.get("trained_policy", {}) or {}).get("n_sessions")

    return {
        "policy_names": ordered_names,
        "reward_values": reward_values,
        "purchase_values": purchase_values,
        "best_policy": best_policy,
        "delta_vs_uniform": f"{delta_vs_uniform:.3f}",
        "lift_display": lift_display,
        "n_sessions": n_sessions,
    }


def _get_timestamp() -> str:
    """Get current timestamp."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate HTML report for training/evaluation results")
    parser.add_argument("--output-dir", default="outputs/", help="Directory for HTML report")
    parser.add_argument("--policy-file", help="Path to trained_policy.json")
    parser.add_argument("--dataset-file", help="Path to dataset_summary.json")
    parser.add_argument("--ope-file", help="Path to ope_results.json")
    parser.add_argument("--sim-file", help="Path to sim_eval_results.json")
    args = parser.parse_args()
    
    report_path = generate_html_report(
        output_dir=args.output_dir,
        policy_path=args.policy_file,
        dataset_path=args.dataset_file,
        ope_path=args.ope_file,
        sim_path=args.sim_file,
    )
    print(f"✓ Report generated: {report_path}")
