"""Utilities for dealing with fetching and manipulating metadata"""

from typing import Optional
import pandas as pd


def load_sheet_url() -> str:
    """Loads sheet url from disk."""
    with open("../dat/sheet_url.txt") as f:
        url = f.read()
    return url


def fetch_metadata(sheet_url: Optional[str] = None) -> pd.DataFrame:
    """Fetches metadata in for of a Pandas Dataframe
    from the supplied Google Sheets URL.
    """
    if sheet_url is None:
        sheet_url = load_sheet_url()
    sheet_url = sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")
    metadata = pd.read_csv(sheet_url)
    return metadata
