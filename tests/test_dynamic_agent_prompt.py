import importlib.util
import re
from pathlib import Path


def _load_dynamic_assistant():
    module_path = (
        Path(__file__).resolve().parents[1] / "src" / "dynamic-agent.py"
    )
    spec = importlib.util.spec_from_file_location("dynamic_agent", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Assistant


def _normalized_instructions() -> str:
    assistant_cls = _load_dynamic_assistant()
    instructions = assistant_cls(resume="sample").instructions
    return re.sub(r"\s+", " ", instructions).strip().lower()


def test_company_context_trigger_after_identity_role_selection() -> None:
    instructions = _normalized_instructions()

    role_classification_idx = instructions.index("role classification step")
    company_context_idx = instructions.index(
        "company context capture (per relevant role)"
    )
    assert company_context_idx > role_classification_idx

    assert (
        "after the candidate identifies their identity role" in instructions
    )
    assert (
        "ask one company-context question before deep diving into authority and impact"
        in instructions
    )


def test_company_context_includes_required_fields_and_new_company_rule() -> None:
    instructions = _normalized_instructions()

    assert (
        "if conversation moves to a role at a different company or organization, capture context again for that new role before deep probing."
        in instructions
    )
    assert "what the company or organization does" in instructions
    assert "customer segment and business model" in instructions
    assert (
        "approximate size/stage (startup, smb, enterprise, public sector, nonprofit)"
        in instructions
    )
    assert (
        "market, regulatory, or operational constraints relevant to that role"
        in instructions
    )
    assert (
        "how success is measured in that function (function-specific kpis)"
        in instructions
    )


def test_non_company_fallback_context_is_explicit() -> None:
    instructions = _normalized_instructions()

    assert "freelance, founder, agency, consulting, academic, or internal platform" in instructions
    assert (
        'ask for equivalent organization, client, or team context instead of forcing "company" wording'
        in instructions
    )


def test_cross_function_guardrails_and_examples_are_present() -> None:
    instructions = _normalized_instructions()

    assert (
        "treat all functions as first-class: engineering, product, design, marketing, sales, customer success, hr/people, operations, finance, accounting, legal, procurement, analytics, and strategy."
        in instructions
    )
    assert (
        "do not assume technical/software framing unless the candidate indicates it."
        in instructions
    )
    assert "marketing: pipeline, cac, retention, brand lift" in instructions
    assert (
        "finance/accounting: close cycle, controls, audit readiness, margin/cash flow, forecast accuracy"
        in instructions
    )
    assert "operations: throughput, sla, quality, process reliability" in instructions
    assert "engineering/product: latency, reliability, roadmap, adoption" in instructions


def test_no_tech_only_mandate_language() -> None:
    instructions = _normalized_instructions()

    assert "only for technical roles" not in instructions
    assert "must be a software role" not in instructions
    assert "tech roles only" not in instructions


def test_transcript_first_mode_guardrails_present() -> None:
    instructions = _normalized_instructions()

    assert "transcript-first mode" in instructions
    assert "this interview is conversation-first and transcript-first." in instructions
    assert "prioritize extracting concrete evidence in natural dialogue" in instructions
    assert (
        "do not output schemas, json, score tables, capability matrices, or structured profile blocks."
        in instructions
    )
    assert "speak naturally as an interviewer, not as a report generator." in instructions


def test_internal_capability_lens_is_hidden() -> None:
    instructions = _normalized_instructions()

    assert "internal capability lens (hidden)" in instructions
    assert "decision-making depth" in instructions
    assert "ownership scope" in instructions
    assert "commercial impact" in instructions
    assert "problem-solving pattern" in instructions
    assert "influence without authority" in instructions
    assert "execution in ambiguity" in instructions
    assert (
        "do not verbalize scores, rubrics, or explicit ratings unless the candidate directly asks for them."
        in instructions
    )


def test_evidence_quality_followups_are_required() -> None:
    instructions = _normalized_instructions()

    assert "if answers are abstract: convert into measurable terms." in instructions
    assert (
        "when a material metric is mentioned, ask one validation follow-up"
        in instructions
    )
    assert (
        "if evidence is missing for a key claim, ask one clarifying question before moving on."
        in instructions
    )


def test_conversational_close_not_structured_output() -> None:
    instructions = _normalized_instructions()

    assert "final conversational recap" in instructions
    assert "close with a short spoken recap in 4-6 lines:" in instructions
    assert "keep it natural and verbal, not formatted." in instructions
    assert "does this reflect your experience accurately?" in instructions

    assert "final positioning synthesis" not in instructions
    assert "produce a concise recruiter-ready narrative structured as:" not in instructions
