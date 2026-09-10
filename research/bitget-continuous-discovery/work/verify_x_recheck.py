# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "statsmodels", "pydantic>=2", "typer", "rich"]
# ///
"""Recalculate artifact equality and exact lock-file provenance after repairs."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

from pydantic import BaseModel

from run_x_volume_control import ControlReport
from x_discovery_results import DiscoveryReport

WORK = Path(__file__).resolve().parent


class Evidence(BaseModel):
    discovery_reports_equal: bool
    control_reports_equal: bool
    compared_npz_arrays: int
    original_lock_byte_hash_matched: bool
    corrected_lock_byte_hash_matches: bool
    original_discovery_sha256: str
    recheck_discovery_sha256: str
    recheck_control_sha256: str


def main() -> None:
    original = WORK / "x_discovery_20260910"
    recheck = WORK / "x_discovery_20260910_recheck"
    first = DiscoveryReport.model_validate_json((original / "report.json").read_bytes())
    final = DiscoveryReport.model_validate_json((recheck / "report.json").read_bytes())
    original_control = WORK / "x_volume_control_20260910"
    recheck_control = WORK / "x_volume_control_20260910_recheck"
    before = ControlReport.model_validate_json((original_control / "report.json").read_bytes())
    after = ControlReport.model_validate_json((recheck_control / "report.json").read_bytes())
    count = 0
    for left, right in ((original, recheck), (original_control, recheck_control)):
        assert sorted(item.name for item in left.glob("*.npz")) == sorted(item.name for item in right.glob("*.npz"))
        for old_file in left.glob("*.npz"):
            with ZipFile(old_file) as old, ZipFile(right / old_file.name) as new:
                assert old.namelist() == new.namelist()
                for key in old.namelist():
                    assert key.endswith(".npy") and old.read(key) == new.read(key)
                    count += 1
    evidence = Evidence(discovery_reports_equal=first == final, control_reports_equal=before == after,
                        compared_npz_arrays=count,
                        original_lock_byte_hash_matched=sha256((original / "test_ids_locked.txt").read_bytes()).hexdigest() == first.test_ids_sha256,
                        corrected_lock_byte_hash_matches=sha256((recheck / "test_ids_locked.txt").read_bytes()).hexdigest() == final.test_ids_sha256,
                        original_discovery_sha256=sha256((original / "report.json").read_bytes()).hexdigest(),
                        recheck_discovery_sha256=sha256((recheck / "report.json").read_bytes()).hexdigest(),
                        recheck_control_sha256=sha256((recheck_control / "report.json").read_bytes()).hexdigest())
    assert evidence.discovery_reports_equal and evidence.control_reports_equal and evidence.corrected_lock_byte_hash_matches
    _ = (WORK / "X_RECHECK_EVIDENCE_20260910.json").write_bytes(evidence.model_dump_json(indent=2).encode())
    print(evidence.model_dump_json())


if __name__ == "__main__":
    main()
