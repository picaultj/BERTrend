#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pandas as pd

from bertrend.bertrend_apps.prospective_demo import (
    LLM_TOPIC_DESCRIPTION_COLUMN,
    LLM_TOPIC_TITLE_COLUMN,
)
from bertrend.bertrend_apps.prospective_demo.topic_search import (
    filter_topics_by_keywords,
)


def _make_df():
    return pd.DataFrame(
        {
            "Topic": [0, 1, 2],
            LLM_TOPIC_TITLE_COLUMN: [
                "Nuclear energy policy",
                "Solar power growth",
                "Electric vehicles market",
            ],
            LLM_TOPIC_DESCRIPTION_COLUMN: [
                "Debate about nuclear reactors and safety.",
                "Photovoltaic installations expand rapidly.",
                "Battery costs fall as EV adoption rises.",
            ],
            "Representation": [
                ["nuclear", "reactor", "energy"],
                ["solar", "photovoltaic", "power"],
                ["battery", "vehicle", "electric"],
            ],
        }
    )


def test_blank_query_returns_unchanged():
    df = _make_df()
    for query in ["", "   ", None]:
        result = filter_topics_by_keywords(df, query)
        assert result.equals(df)


def test_none_and_empty_df():
    assert filter_topics_by_keywords(None, "nuclear") is None
    empty = pd.DataFrame()
    assert filter_topics_by_keywords(empty, "nuclear").empty


def test_case_insensitive_title_match():
    df = _make_df()
    result = filter_topics_by_keywords(df, "NUCLEAR")
    assert result["Topic"].tolist() == [0]


def test_match_on_description():
    df = _make_df()
    result = filter_topics_by_keywords(df, "photovoltaic")
    assert result["Topic"].tolist() == [1]


def test_match_on_representation():
    df = _make_df()
    result = filter_topics_by_keywords(df, "battery")
    assert result["Topic"].tolist() == [2]


def test_multiple_terms_are_anded():
    df = _make_df()
    # Both terms appear only for topic 0 (title + description)
    result = filter_topics_by_keywords(df, "nuclear safety")
    assert result["Topic"].tolist() == [0]
    # "nuclear" and "solar" never co-occur in a single row
    assert filter_topics_by_keywords(df, "nuclear solar").empty


def test_no_match_returns_empty():
    df = _make_df()
    assert filter_topics_by_keywords(df, "aviation").empty


def test_missing_search_columns_returns_unchanged():
    df = pd.DataFrame({"Topic": [0, 1], "Other": ["a", "b"]})
    result = filter_topics_by_keywords(df, "nuclear")
    assert result.equals(df)


def test_custom_search_columns():
    df = _make_df()
    # Restrict search to the title only: "reactors" lives in the description,
    # so it should not match when only the title is searched.
    result = filter_topics_by_keywords(
        df, "reactors", search_columns=[LLM_TOPIC_TITLE_COLUMN]
    )
    assert result.empty
