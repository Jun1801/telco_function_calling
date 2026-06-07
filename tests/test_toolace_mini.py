from src.generation.toolace_mini import TelcoToolACEMiniPipeline


def test_toolace_mini_pipeline_generates_verified_samples() -> None:
    samples = TelcoToolACEMiniPipeline("data").generate_verified_samples()

    assert 50 <= len(samples) <= 100
    assert all(sample["source"] == "telco_toolace_mini" for sample in samples)
    assert all(sample["generator"] == "telco_toolace_mini" for sample in samples)
    assert all("toolace_validation" in sample for sample in samples)
    assert all("metrics" in sample["toolace_validation"] for sample in samples)
    assert all("scenario_family" in sample for sample in samples)
    assert all("generation_profile" in sample for sample in samples)
    assert any(sample["scenario"] == "function_name_masking" for sample in samples)
    assert any(sample["scenario"] == "contract_violation" for sample in samples)
    assert any(sample["scenario_family"] == "masking" for sample in samples)
