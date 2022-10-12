"""Utilities for dealing with fetching and manipulating metadata"""

import pandas as pd

DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/181pbNCULuYKO5yPrWIfdmOLkrps2WUpfoJ8mtT56vuw/edit#gid=1762774185"


def fetch_metadata(sheet_url: str = DEFAULT_SHEET_URL) -> pd.DataFrame:
    """Fetches metadata in for of a Pandas Dataframe
    from the supplied Google Sheets URL.
    """
    sheet_url = sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")
    metadata = pd.read_csv(sheet_url)
    return metadata
