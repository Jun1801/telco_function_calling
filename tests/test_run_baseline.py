import json
import subprocess
import sys
from pathlib import Path


def test_run_baseline_mock_oracle_writes_reports(tmp_path: Path) -> None:
    output = tmp_path / "prompt_only_results.jsonl"
    error_report = tmp_path / "error_analysis.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_baseline.py",
            "--output",
            str(output),
            "--error-report",
            str(error_report),
        ],
        check=True,
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records
    assert error_report.exists()
    assert all("prompt" in record for record in records)
    assert all("reward" in record for record in records)
