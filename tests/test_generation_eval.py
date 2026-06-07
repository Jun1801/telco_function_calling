from pathlib import Path

from src.generation.synth_data_generator import build_samples, write_jsonl_by_split


def test_generator_writes_expected_splits(tmp_path: Path) -> None:
    counts = write_jsonl_by_split(build_samples(), tmp_path)

    assert 50 <= sum(counts.values()) <= 100
    assert counts["train"] >= 10
    assert counts["eval_seen"] >= 5
    assert counts["eval_contract"] >= 5
    assert counts["eval_missing_slot"] >= 5
    assert counts["eval_abstention"] >= 4
    assert counts["eval_masked_tools"] >= 5
    assert counts["train"] >= 35
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "eval_evolution_deprecated.jsonl").exists()
    assert (tmp_path / "eval_expanded_library.jsonl").exists()
