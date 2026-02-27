from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(script_name: str) -> None:
    script_path = ROOT / "scripts" / script_name
    cmd = [sys.executable, str(script_path)]
    print(f"\n=== Running {script_name} ===")
    subprocess.run(cmd, check=True)
    print("[OK]")


def main() -> None:
    run_step("build_dpi_ready_index.py")
    run_step("build_doughnut_subindex.py")
    run_step("build_trend_analysis.py")
    print("\nCore Python pipeline complete.")


if __name__ == "__main__":
    main()
