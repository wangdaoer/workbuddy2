"""Pipeline wrapper for the incumbent point-in-time amount-capacity stress.

The wrapped script (`run_incumbent_capacity_stress.py`) is research-only: it
produces a capacity audit (largest acceptable capital under amount-participation
limits) and never changes the production ranking, weights, or execution config.

This wrapper is best-effort so the capacity check can NEVER abort the daily
pipeline chain (MOS section 6: the daily run must stay whole-chain consistent
and idempotent). Any failure is reported as a warning on stderr and the process
exits 0. The run card will then show `no_data` for this step, while the warning
artifact below records the actual reason for inspection.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import run_incumbent_capacity_stress as ics  # noqa: E402


def main() -> None:
    try:
        ics.main()
    except Exception as exc:  # noqa: BLE001 - best-effort: never break the chain
        print(
            f"CAPACITY_STRESS_WARNING: incumbent capacity stress skipped: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        # Write a warning artifact so the skipped reason is inspectable even
        # though the run card records this step as no_data.
        try:
            args = ics.parse_args()
            warning_dir = Path(args.output_dir)
            warning_dir.mkdir(parents=True, exist_ok=True)
            (warning_dir / "capacity_stress_warning.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
        except Exception:  # pragma: no cover - secondary best-effort
            pass
        # best-effort: never abort the pipeline
        return


if __name__ == "__main__":
    main()
