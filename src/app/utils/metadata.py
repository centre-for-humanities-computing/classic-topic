"""Utilities for dealing with fetching and manipulating metadata"""

import pandas as pd

DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/15WIzk2aV3vCQLnDihdnNCLxMbDmJZiZKmuiM_xRKbwk/edit#gid=282554525"


def fetch_metadata(sheet_url: str = DEFAULT_SHEET_URL) -> pd.DataFrame:
    """Fetches metadata in for of a Pandas Dataframe
    from the supplied Google Sheets URL.
    """
    sheet_url = sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")
    metadata = pd.read_csv(sheet_url)
    return metadata
