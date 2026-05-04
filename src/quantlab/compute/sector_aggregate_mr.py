"""MapReduce job: per-sector aggregate trading volume across many tickers.

Implemented with :mod:`mrjob` (W9 material). The mapper emits
``(sector, volume)`` for every row, and the reducer sums volume per sector.

Usage from the shell::

    python -m quantlab.compute.sector_aggregate_mr prices_with_sector.csv

The script also exposes :func:`run_inline` for use from Python or unit tests
without spawning a subprocess.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from collections.abc import Iterable

from mrjob.job import MRJob
from mrjob.step import MRStep


class SectorVolumeMR(MRJob):
    """MapReduce job summing daily trading volume per sector."""

    def steps(self) -> list[MRStep]:
        return [MRStep(mapper=self.mapper_emit_sector_volume,
                       reducer=self.reducer_sum_volume)]

    def mapper_emit_sector_volume(self, _, line: str):
        if line.startswith("date,") or not line.strip():
            return
        parts = next(csv.reader([line]))
        try:
            sector = parts[2]
            volume = float(parts[3])
        except (IndexError, ValueError):
            return
        yield sector, volume

    def reducer_sum_volume(self, sector: str, volumes: Iterable[float]):
        yield sector, sum(volumes)


def run_inline(rows: Iterable[tuple[str, str, str, float]]) -> dict[str, float]:
    """Pure-Python equivalent of the MR job.

    Used by unit tests so we don't need to spawn the mrjob subprocess
    machinery to verify the aggregation logic. The mapper / reducer methods
    above are still exercised in tests via direct invocation.
    """
    totals: dict[str, float] = defaultdict(float)
    for _date, _ticker, sector, volume in rows:
        totals[sector] += float(volume)
    return dict(totals)


def emit_csv(rows: Iterable[tuple[str, str, str, float]]) -> str:
    """Render ``(date, ticker, sector, volume)`` rows as CSV (used by tests)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["date", "ticker", "sector", "volume"])
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


if __name__ == "__main__":  # pragma: no cover
    SectorVolumeMR.run()
