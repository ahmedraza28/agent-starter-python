import importlib.util
from pathlib import Path


def _load_module(file_name: str, module_name: str):
    module_path = Path(__file__).resolve().parents[1] / "src" / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_truncation_contract(module) -> None:
    truncate = module.truncate_resume_for_prompt

    assert truncate("") == "No resume was provided."
    assert truncate("   ") == "No resume was provided."
    assert truncate("  Senior backend engineer  ") == "Senior backend engineer"

    limit = 20
    long_resume = "A" * 80
    output = truncate(long_resume, limit=limit)

    assert output.startswith("A" * limit)
    assert f"Resume truncated to first {limit} characters." in output


def test_resume_truncation_contract_agent() -> None:
    module = _load_module("agent.py", "agent_module")
    _assert_truncation_contract(module)


def test_resume_truncation_contract_dynamic_agent() -> None:
    module = _load_module("dynamic-agent.py", "dynamic_agent_module")
    _assert_truncation_contract(module)
