"""Scrape the S&P 500 constituent list from Wikipedia.

Wikipedia's table layout is far more stable than Yahoo Finance's HTML, which
makes this an honest BeautifulSoup demo (W11–W12 material) without the
brittleness of scraping a JavaScript-heavy site.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import requests
from bs4 import BeautifulSoup

from quantlab._logging import get_logger
from quantlab.exceptions import DataSourceError

log = get_logger(__name__)

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

Fetcher = Callable[[str], str]


@dataclass(frozen=True)
class Constituent:
    """A single S&P 500 row, as parsed from Wikipedia."""

    symbol: str
    security: str
    sector: str
    sub_industry: str


def _default_fetcher(url: str) -> str:
    """Fetch ``url`` with a polite User-Agent. Tests inject a stub instead."""
    response = requests.get(url, headers={"User-Agent": "quantlab/0.1"}, timeout=15)
    response.raise_for_status()
    return response.text


class WikipediaConstituents:
    """Parse the S&P 500 constituent table from a Wikipedia HTML document.

    Args:
        fetcher: Optional callable that returns the page HTML for a URL.
            Defaults to ``requests.get``; tests pass a callable that reads a
            local fixture so CI never hits the network.
    """

    def __init__(self, fetcher: Fetcher | None = None) -> None:
        self._fetcher = fetcher or _default_fetcher

    def fetch(self, url: str = WIKI_SP500_URL) -> pd.DataFrame:
        """Return a long-format DataFrame of S&P 500 constituents.

        Returns:
            DataFrame with columns ``symbol``, ``security``, ``sector``,
            ``sub_industry``.
        """
        log.info("fetching S&P 500 constituents from %s", url)
        try:
            html = self._fetcher(url)
        except Exception as exc:
            raise DataSourceError(f"failed to fetch {url}: {exc}") from exc
        return self.parse(html)

    @staticmethod
    def parse(html: str) -> pd.DataFrame:
        """Parse Wikipedia constituent HTML into a tidy DataFrame.

        Looks for the first ``wikitable sortable`` table on the page.
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", attrs={"class": lambda c: c and "wikitable" in c})
        if table is None:
            raise DataSourceError("no wikitable found in document")

        rows: list[Constituent] = []
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if len(cells) < 4:
                continue
            rows.append(
                Constituent(
                    symbol=cells[0].replace(".", "-"),
                    security=cells[1],
                    sector=cells[2],
                    sub_industry=cells[3],
                )
            )

        if not rows:
            raise DataSourceError("wikitable contained no parseable rows")

        df = pd.DataFrame([c.__dict__ for c in rows])
        return df.sort_values("symbol").reset_index(drop=True)
