#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

"""Keyword search/filtering over topic dataframes for the prospective demo."""

import pandas as pd

from bertrend.bertrend_apps.prospective_demo import (
    LLM_TOPIC_DESCRIPTION_COLUMN,
    LLM_TOPIC_TITLE_COLUMN,
)

# Columns scanned by the keyword search: the LLM-generated topic title and
# description, plus the raw topic-word representation when it is available.
TOPIC_SEARCH_COLUMNS = [
    LLM_TOPIC_TITLE_COLUMN,
    LLM_TOPIC_DESCRIPTION_COLUMN,
    "Representation",
]


def filter_topics_by_keywords(
    df: pd.DataFrame,
    query: str,
    search_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Filter a topics dataframe to the rows matching a keyword query.

    Matching is case-insensitive substring containment across the topic title,
    LLM description and representation columns. Whitespace-separated terms are
    combined with AND semantics: a row is kept only if every term appears
    somewhere in the searched columns. A ``None``/blank query (or an empty
    dataframe) returns the input unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        The topics dataframe to filter.
    query : str
        The search string; whitespace splits it into individual terms.
    search_columns : list[str], optional
        Columns to search. Defaults to :data:`TOPIC_SEARCH_COLUMNS`. Columns
        absent from ``df`` are ignored.

    Returns
    -------
    pd.DataFrame
        The filtered dataframe (a view/copy of matching rows), or ``df``
        unchanged when the query is blank or nothing can be searched.
    """
    if df is None or df.empty or not query or not query.strip():
        return df

    cols = search_columns if search_columns is not None else TOPIC_SEARCH_COLUMNS
    present_cols = [c for c in cols if c in df.columns]
    if not present_cols:
        return df

    # Build one lowercase searchable string per row from the available columns.
    haystack = df[present_cols].astype(str).agg(" ".join, axis=1).str.lower()

    mask = pd.Series(True, index=df.index)
    for term in query.lower().split():
        mask &= haystack.str.contains(term, regex=False, na=False)

    return df[mask]
