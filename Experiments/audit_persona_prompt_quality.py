#!/usr/bin/env python3
"""LLM-as-a-judge evaluation of the five Clickworker scenario prompts.

The participant-facing scenario wording is an experimental instrument: it supplies
the situational motive that distinguishes the five persona cells while having to
stay neutral about what participants should *do*. This script scores that wording
against a fixed rubric so the choice of final wording is evidenced rather than
asserted, as required for the preregistration.

Two properties make the result defensible:

* **The audited text is the deployed text.** Scenarios are parsed out of the Jinja
  AST of ``DemoSiteV{2,3}/app/templates/home.html`` rather than copied into this
  file, so the audit cannot silently drift from what participants actually see.
  Both shops are asserted to carry identical wording.
* **Revisions are scored blind and in randomised order.** The judge is never told
  which revision it is looking at, and persona order is reshuffled per replicate,
  so neither revision identity nor position can drive the scores.

Four criteria are scored per persona; cross-cell fairness is a property of the set
and is therefore scored once over all five together.

Usage::

    pip install -r Experiments/requirements.txt

    # Check the deployed wording still matches the preregistration lock.
    # No API calls, no credentials — safe to run in CI.
    python audit_persona_prompt_quality.py --verify-lock

    # Score the current working-tree wording against the pre-softening baseline
    python audit_persona_prompt_quality.py --replicates 5

    # Current wording only
    python audit_persona_prompt_quality.py --revisions working

Scoring requires ``ANTHROPIC_API_KEY`` (or an ``ant auth login`` profile);
``--verify-lock`` does not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from jinja2 import Environment, nodes
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = "DemoSiteV3/app/templates/home.html"
MIRROR_TEMPLATE_PATH = "DemoSiteV2/app/templates/home.html"

# Judge model. Pinned in the thesis and the preregistration — changing it
# invalidates the reported scores and requires a re-run.
JUDGE_MODEL = "claude-opus-5"

PERSONA_ORDER = [
    "explorer",
    "fastbuyer",
    "detailedcomparator",
    "discounthunter",
    "windowshopper",
]

# The revision the scenario wording was softened in (Stefan's feedback: the
# original wording prescribed behavior and would have forced the archetypes
# rather than letting them emerge). Its parent is the pre-softening baseline.
SOFTENING_COMMIT = "f9c6392"


# ── Rubric ───────────────────────────────────────────────────────────────────
# Criterion definitions are part of the published methodology. Edit with care:
# the rubric hash is recorded in the output and cited in the appendix.

STUDY_CONTEXT = """\
The prompts under evaluation are shown once, in a modal, to crowdworkers who have \
just entered a simulated online shop built for a research study. Each participant \
sees exactly one of the five prompts and then browses freely; the session ends when \
they choose to end it. No real money is involved and no real order is placed.

The five prompts are meant to induce five different shopping dispositions \
(open-ended browsing; a specific electronics need; a considered home purchase; \
constrained budget; idle time with no goal) purely by supplying a situational \
motive.\
"""

HYPOTHESES_WITHHELD_FROM_PARTICIPANTS = """\
Participants must NOT be able to infer any of the following from the prompt:
1. That the shop displays adaptive recommendation widgets whose selection varies.
2. That there are two experimental conditions using different decision policies.
3. That their behavior will be compared against a behavioral simulation.
4. That purchasing, conversion, or funnel depth are outcomes of interest.
5. That they have been assigned a persona or behavioral archetype.\
"""

CRITERIA: dict[str, str] = {
    "clarity": (
        "Is the prompt immediately understandable to a non-expert crowdworker on a "
        "single reading? Consider sentence complexity, ambiguity, and whether the "
        "participant would know what to do next. A prompt that requires re-reading, "
        "or that leaves the participant unsure whether they may leave, scores low."
    ),
    "behavioral_specificity": (
        "Does the prompt establish a distinct situational motive that separates this "
        "cell from the other four, WITHOUT prescribing actions? The target is a "
        "motive ('money is tight this month'), not an instruction ('compare at least "
        "three products'). Score low both when the motive is too vague to "
        "differentiate the cell AND when the prompt dictates the behavior it is "
        "supposed to elicit — the latter manufactures the archetype instead of "
        "observing it."
    ),
    "neutrality": (
        "Is the prompt free of demand characteristics — cues that tell the "
        "participant which behavior the researcher wants? Look for implied "
        "obligations to purchase, to spend a particular amount of time, to explore "
        "a certain number of pages, or any framing that makes one course of action "
        "read as the 'correct' one. Explicit permission not to act is neutral; "
        "pressure toward action is not."
    ),
    "hypothesis_leakage": (
        "Could an attentive participant infer any of the withheld facts listed above "
        "from this prompt? Score 5 when nothing in the wording points to them, and 1 "
        "when the prompt effectively announces one. Judge only what the prompt "
        "reveals, not what participants might notice later while browsing."
    ),
}

CROSS_CELL_CRITERION = (
    "Cross-cell fairness: are these five prompts matched in length, register, "
    "reading difficulty, and motivational intensity, so that differences in "
    "observed behavior can be attributed to the situational motive rather than to "
    "the prompts being unequally demanding or unequally engaging? Systematic "
    "asymmetry — one prompt notably longer, more urgent, or more prescriptive than "
    "the rest — is the failure mode. Score 5 for a well-matched set, 1 when at "
    "least one cell is clearly advantaged or disadvantaged by its wording."
)

SCALE = (
    "Score each criterion on an integer scale from 1 to 5: "
    "1 = serious problem that would compromise the study, "
    "2 = clear weakness, "
    "3 = acceptable but improvable, "
    "4 = good with only minor reservations, "
    "5 = no reservations. "
    "Use the full range; do not default to the middle."
)

JUDGE_SYSTEM = f"""\
You are an experimental-methods reviewer evaluating participant-facing scenario \
wording for a within-subjects online shopping study. You are rigorous and \
independent: your task is to find weaknesses, not to endorse the material.

# Study context
{STUDY_CONTEXT}

# Facts that must not leak to participants
{HYPOTHESES_WITHHELD_FROM_PARTICIPANTS}

# Scoring scale
{SCALE}

Judge only the text placed in front of you. Do not speculate about material you \
have not been shown, and do not reward a prompt for being short if brevity costs \
clarity. Each justification must cite specific wording from the prompt.\
"""


class CriterionScore(BaseModel):
    """One criterion's verdict for one prompt."""

    score: int = Field(ge=1, le=5, description="Integer 1-5 per the scale.")
    justification: str = Field(
        description="Two or three sentences citing specific wording from the prompt."
    )


class PersonaVerdict(BaseModel):
    """Per-persona scores across the four prompt-level criteria."""

    clarity: CriterionScore
    behavioral_specificity: CriterionScore
    neutrality: CriterionScore
    hypothesis_leakage: CriterionScore
    strongest_concern: str = Field(
        description="The single most consequential weakness, or 'none' if there is none."
    )


class CrossCellVerdict(BaseModel):
    """Set-level fairness verdict over all five prompts together."""

    fairness: CriterionScore
    least_comparable_cell: str = Field(
        description="Label of the prompt that fits the set least well, or 'none'."
    )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_scenarios(template_source: str) -> dict[str, dict[str, str]]:
    """Pull the ``_scenarios`` literal out of a home.html Jinja template.

    Parsing the template AST rather than re-declaring the prompts here is what
    guarantees the audited wording is the wording that ships.

    Args:
        template_source: Full text of a ``home.html`` template.

    Returns:
        Mapping of persona key to its ``title``/``body``/``note`` dict.

    Raises:
        ValueError: If the template contains no ``_scenarios`` assignment.
    """
    ast = Environment().parse(template_source)
    for node in ast.find_all(nodes.Assign):
        if isinstance(node.target, nodes.Name) and node.target.name == "_scenarios":
            return node.node.as_const()
    raise ValueError("no `_scenarios` assignment found in template")


def load_revision(revision: str) -> dict[str, dict[str, str]]:
    """Load the scenario set as of a given git revision (or the working tree).

    Args:
        revision: ``"working"`` for the current files, otherwise any git revision.

    Returns:
        Mapping of persona key to its scenario dict.
    """
    if revision == "working":
        source = (REPO_ROOT / TEMPLATE_PATH).read_text(encoding="utf-8")
        mirror = (REPO_ROOT / MIRROR_TEMPLATE_PATH).read_text(encoding="utf-8")
    else:
        source = subprocess.check_output(
            ["git", "show", f"{revision}:{TEMPLATE_PATH}"], cwd=REPO_ROOT, text=True
        )
        mirror = subprocess.check_output(
            ["git", "show", f"{revision}:{MIRROR_TEMPLATE_PATH}"], cwd=REPO_ROOT, text=True
        )

    scenarios = extract_scenarios(source)
    mirror_scenarios = extract_scenarios(mirror)
    if scenarios != mirror_scenarios:
        raise ValueError(
            f"V2 and V3 scenario wording differs at revision {revision!r}; the two "
            "conditions must present identical prompts."
        )
    return scenarios


def render_prompt(scenario: dict[str, str]) -> str:
    """Render one scenario exactly as the participant reads it in the modal."""
    return f"{scenario['title']}\n\n{scenario['body']}\n\n{scenario['note']}"


def _client():
    import anthropic

    return anthropic.Anthropic()


def _system_blocks() -> list[dict]:
    """System prompt as a cacheable block — identical across every judgment."""
    return [
        {
            "type": "text",
            "text": JUDGE_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _check_refusal(response, what: str) -> None:
    if response.stop_reason == "refusal":
        detail = getattr(response, "stop_details", None)
        raise RuntimeError(
            f"judge declined to score {what} "
            f"(category={getattr(detail, 'category', None)!r}). "
            "Re-run; if it persists the rubric wording needs adjusting."
        )


def score_persona(client, label: str, prompt_text: str) -> PersonaVerdict:
    """Score one prompt on the four prompt-level criteria.

    Args:
        client: Anthropic client.
        label: Persona key, used only for error messages — withheld from the judge
            so the archetype name cannot prime the verdict.
        prompt_text: The rendered participant-facing prompt.

    Returns:
        The parsed per-criterion verdict.
    """
    criteria_block = "\n\n".join(
        f"## {name}\n{definition}" for name, definition in CRITERIA.items()
    )
    user = (
        f"Evaluate the following participant-facing scenario prompt.\n\n"
        f"<prompt>\n{prompt_text}\n</prompt>\n\n"
        f"Score it on each of these four criteria:\n\n{criteria_block}"
    )
    response = client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=16000,
        system=_system_blocks(),
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": user}],
        output_format=PersonaVerdict,
    )
    _check_refusal(response, f"persona {label!r}")
    return response.parsed_output


def score_cross_cell(client, rendered: list[tuple[str, str]]) -> CrossCellVerdict:
    """Score the five prompts together for cross-cell fairness.

    Args:
        client: Anthropic client.
        rendered: ``(anonymous_label, prompt_text)`` pairs in presentation order.

    Returns:
        The parsed set-level verdict.
    """
    block = "\n\n".join(
        f"<prompt label=\"{label}\">\n{text}\n</prompt>" for label, text in rendered
    )
    user = (
        "Below are all five scenario prompts used in the study — one per "
        "experimental cell. Evaluate them as a set.\n\n"
        f"{block}\n\n## fairness\n{CROSS_CELL_CRITERION}"
    )
    response = client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=16000,
        system=_system_blocks(),
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": user}],
        output_format=CrossCellVerdict,
    )
    _check_refusal(response, "the prompt set")
    return response.parsed_output


def evaluate_revision(
    client, revision: str, scenarios: dict, replicates: int, rng: random.Random
) -> dict:
    """Run all replicates for one revision.

    Persona order is reshuffled per replicate so presentation position cannot
    systematically favour a cell, and the judge is shown neutral labels rather
    than archetype names.

    Args:
        client: Anthropic client.
        revision: Revision identifier, recorded in the output.
        scenarios: Persona key to scenario dict.
        replicates: Number of independent judging passes.
        rng: Seeded RNG driving the shuffles.

    Returns:
        A dict of per-persona scores, cross-cell scores, and provenance.
    """
    per_persona: dict[str, dict[str, list[int]]] = {
        key: {criterion: [] for criterion in CRITERIA} for key in scenarios
    }
    justifications: dict[str, list[dict]] = {key: [] for key in scenarios}
    fairness_scores: list[int] = []
    fairness_notes: list[dict] = []

    for replicate in range(replicates):
        order = list(scenarios)
        rng.shuffle(order)

        for key in order:
            text = render_prompt(scenarios[key])
            verdict = score_persona(client, key, text)
            for criterion in CRITERIA:
                per_persona[key][criterion].append(getattr(verdict, criterion).score)
            justifications[key].append(
                {
                    "replicate": replicate,
                    "strongest_concern": verdict.strongest_concern,
                    **{
                        criterion: getattr(verdict, criterion).justification
                        for criterion in CRITERIA
                    },
                }
            )

        # Neutral labels: the judge must not be able to map a prompt back to an
        # archetype name when assessing whether the set is balanced.
        rendered = [(f"cell-{i + 1}", render_prompt(scenarios[k])) for i, k in enumerate(order)]
        cross = score_cross_cell(client, rendered)
        fairness_scores.append(cross.fairness.score)
        fairness_notes.append(
            {
                "replicate": replicate,
                "presentation_order": order,
                "least_comparable_cell": cross.least_comparable_cell,
                "justification": cross.fairness.justification,
            }
        )

    def summarise(values: list[int]) -> dict:
        return {
            "mean": round(statistics.fmean(values), 3),
            "sd": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "n": len(values),
            "raw": values,
        }

    return {
        "revision": revision,
        "prompt_sha256": {
            key: _sha(render_prompt(scenario)) for key, scenario in scenarios.items()
        },
        "prompts": {key: render_prompt(scenario) for key, scenario in scenarios.items()},
        "per_persona": {
            key: {criterion: summarise(scores) for criterion, scores in criteria.items()}
            for key, criteria in per_persona.items()
        },
        "cross_cell_fairness": summarise(fairness_scores),
        "justifications": justifications,
        "fairness_notes": fairness_notes,
    }


def overall_mean(result: dict) -> float:
    """Mean of every prompt-level criterion score across all personas."""
    values = [
        summary["mean"]
        for criteria in result["per_persona"].values()
        for summary in criteria.values()
    ]
    return round(statistics.fmean(values), 3)


def write_latex_table(results: list[dict], path: Path) -> None:
    """Emit the appendix table comparing revisions per criterion."""
    criteria = list(CRITERIA)
    lines = [
        "% Generated by Experiments/audit_persona_prompt_quality.py -- do not edit by hand.",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{LLM-as-a-judge scores for the Clickworker scenario wording, mean over "
        "replicates (1--5 scale, higher is better).}",
        "\\label{tab:persona-judge}",
        "\\begin{tabular}{l" + "c" * (len(criteria) + 1) + "}",
        "\\toprule",
        "\\textbf{Revision} & "
        + " & ".join("\\textbf{" + c.replace("_", " ").title() + "}" for c in criteria)
        + " & \\textbf{Fairness} \\\\",
        "\\midrule",
    ]
    for result in results:
        per_criterion = []
        for criterion in criteria:
            means = [
                result["per_persona"][key][criterion]["mean"] for key in result["per_persona"]
            ]
            per_criterion.append(f"{statistics.fmean(means):.2f}")
        lines.append(
            f"{result['revision']} & "
            + " & ".join(per_criterion)
            + f" & {result['cross_cell_fairness']['mean']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(results: list[dict], path: Path) -> None:
    """Emit the flat per-persona/per-criterion score table."""
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["revision", "persona", "criterion", "mean", "sd", "min", "max", "n"])
        for result in results:
            for persona, criteria in result["per_persona"].items():
                for criterion, summary in criteria.items():
                    writer.writerow(
                        [
                            result["revision"],
                            persona,
                            criterion,
                            summary["mean"],
                            summary["sd"],
                            summary["min"],
                            summary["max"],
                            summary["n"],
                        ]
                    )
            fairness = result["cross_cell_fairness"]
            writer.writerow(
                [
                    result["revision"],
                    "ALL",
                    "cross_cell_fairness",
                    fairness["mean"],
                    fairness["sd"],
                    fairness["min"],
                    fairness["max"],
                    fairness["n"],
                ]
            )


# Digests of the wording locked in ClickworkerPreregistration.md Section 2.1.
# `--verify-lock` fails if the deployed template no longer matches, so the
# re-lock rule is enforced by CI rather than by remembering.
LOCKED_DIGESTS: dict[str, str] = {
    "explorer": "e45a0def8c86b980",
    "fastbuyer": "e1189cee0cc54c21",
    "detailedcomparator": "6ff75714e313aac4",
    "discounthunter": "ed56d5b61eee5d6a",
    "windowshopper": "a476202b552266a5",
}
LOCKED_SET_DIGEST = "b94ced0ca08ba763836db5e4d7eb574265022fd6065b3d3402d0cb83738eff77"


def verify_lock() -> int:
    """Check the deployed wording against the preregistered digests.

    Returns:
        0 when every prompt matches the lock, 1 otherwise.
    """
    scenarios = load_revision("working")
    drift = []
    for key in PERSONA_ORDER:
        actual = _sha(render_prompt(scenarios[key]))[:16]
        expected = LOCKED_DIGESTS[key]
        status = "ok" if actual == expected else "DRIFTED"
        if actual != expected:
            drift.append(key)
        print(f"  {key:<20} {actual}  {status}")

    joined = "\n\x00\n".join(render_prompt(scenarios[k]) for k in PERSONA_ORDER)
    set_digest = _sha(joined)
    set_ok = set_digest == LOCKED_SET_DIGEST
    print(f"  {'set digest':<20} {set_digest[:16]}  {'ok' if set_ok else 'DRIFTED'}")

    if drift or not set_ok:
        print(
            "\nScenario wording no longer matches the preregistration lock "
            f"({', '.join(drift) or 'set digest'}).\n"
            "Re-run the judge audit, update Section 2.1 of "
            "ClickworkerPreregistration.md and LOCKED_DIGESTS here, and record the "
            "change as a preregistration revision before recruiting.",
            file=sys.stderr,
        )
        return 1
    print("\nWording matches the preregistration lock.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-lock",
        action="store_true",
        help="Check deployed wording against the preregistered digests and exit "
        "(no API calls, no credentials needed).",
    )
    parser.add_argument(
        "--revisions",
        default=f"{SOFTENING_COMMIT}^,working",
        help="Comma-separated git revisions to score; 'working' means the current files.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=5,
        help="Independent judging passes per revision (default 5).",
    )
    parser.add_argument("--seed", type=int, default=20260731, help="Shuffle seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "Experiments" / "persona_prompt_quality.json",
        help="Output JSON path; the CSV and LaTeX table sit alongside it.",
    )
    args = parser.parse_args()

    if args.verify_lock:
        return verify_lock()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set.\n"
            "Set it, or run `ant auth login` and re-run — the SDK picks up the profile "
            "automatically.",
            file=sys.stderr,
        )
        return 2

    client = _client()
    rng = random.Random(args.seed)
    revisions = [r.strip() for r in args.revisions.split(",") if r.strip()]

    results = []
    for revision in revisions:
        scenarios = load_revision(revision)
        missing = set(PERSONA_ORDER) - set(scenarios)
        if missing:
            raise ValueError(f"revision {revision!r} is missing personas: {sorted(missing)}")
        print(
            f"Scoring {revision} — {len(scenarios)} prompts x {args.replicates} replicates",
            file=sys.stderr,
        )
        results.append(evaluate_revision(client, revision, scenarios, args.replicates, rng))

    payload = {
        "judge_model": JUDGE_MODEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "replicates": args.replicates,
        "seed": args.seed,
        "rubric_sha256": _sha(JUDGE_SYSTEM + json.dumps(CRITERIA, sort_keys=True) + CROSS_CELL_CRITERION),
        "criteria": CRITERIA,
        "cross_cell_criterion": CROSS_CELL_CRITERION,
        "judge_system_prompt": JUDGE_SYSTEM,
        "results": results,
    }

    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(results, args.out.with_suffix(".csv"))
    write_latex_table(results, args.out.with_name(args.out.stem + "_table.tex"))

    print(f"\nJudge model: {JUDGE_MODEL}  |  replicates: {args.replicates}")
    for result in results:
        print(
            f"  {result['revision']:<12} overall {overall_mean(result):.2f}"
            f"   fairness {result['cross_cell_fairness']['mean']:.2f}"
        )
    print(f"\nWrote {args.out}, {args.out.with_suffix('.csv')}, "
          f"{args.out.with_name(args.out.stem + '_table.tex')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
