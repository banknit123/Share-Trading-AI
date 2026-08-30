from __future__ import annotations

"""Quarantine/reset invalid v31 forward-learning files.

Moves the current candidate/outcome/summary files into a timestamped archive under
`data/live_v31/archive_invalid_*` and leaves a clean ledger for v31.1.
No market data or trading state outside v31 is touched.
"""

from datetime import datetime, timezone
from pathlib import Path
import shutil

ROOT = Path("data/live_v31")
FILES = [
    ROOT / "candidates.csv",
    ROOT / "resolved_outcomes.csv",
    ROOT / "learning_summary.json",
]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = ROOT / f"archive_invalid_{stamp}"
    archive.mkdir(parents=True, exist_ok=True)

    moved = 0
    print("Share-Trading-AI v31 Invalid Forward-Ledger Quarantine")
    for src in FILES:
        if src.exists():
            dst = archive / src.name
            shutil.move(str(src), str(dst))
            print(f"  QUARANTINED {src} -> {dst}")
            moved += 1
        else:
            print(f"  NOT PRESENT {src}")

    note = archive / "README.txt"
    note.write_text(
        "These v31 files were quarantined because the resolution-integrity audit showed\n"
        "that candidate observations were recorded after their historical entry/exit bars\n"
        "were already known. They must not be used as forward-performance evidence.\n",
        encoding="utf-8",
    )

    print(f"\nQuarantined files: {moved}")
    print("Clean v31 ledger is ready for v31.1 fresh-data collection.")
    print(f"Archive retained at: {archive}")


if __name__ == "__main__":
    main()
