#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pandas as pd

from bertrend.bertrend_apps.prospective_demo import (
    LLM_TOPIC_TITLE_COLUMN,
    NOISE,
    STRONG_SIGNALS,
    WEAK_SIGNALS,
)
from bertrend.bertrend_apps.prospective_demo.signal_characterization import (
    IMPACT_COLUMN,
    UNCERTAINTY_COLUMN,
    compute_signal_characterization,
    plot_signal_characterization,
)


def _df(topics, pop, docs, div, titles=None):
    data = {
        "Topic": topics,
        "Latest_Popularity": pop,
        "Docs_Count": docs,
        "Source_Diversity": div,
    }
    if titles is not None:
        data[LLM_TOPIC_TITLE_COLUMN] = titles
    return pd.DataFrame(data)


def test_empty_input_returns_empty_with_columns():
    result = compute_signal_characterization({})
    assert result.empty
    assert IMPACT_COLUMN in result.columns
    assert UNCERTAINTY_COLUMN in result.columns


def test_scores_are_normalized_to_unit_interval():
    dfs = {
        WEAK_SIGNALS: _df([0, 1], [5, 20], [2, 8], [1, 4]),
        STRONG_SIGNALS: _df([2], [40], [30], [10]),
        NOISE: pd.DataFrame(),
    }
    result = compute_signal_characterization(dfs)
    assert len(result) == 3
    assert result[IMPACT_COLUMN].between(0, 1).all()
    assert result[UNCERTAINTY_COLUMN].between(0, 1).all()


def test_higher_popularity_means_higher_impact():
    dfs = {WEAK_SIGNALS: _df([0, 1], [1, 100], [5, 5], [5, 5])}
    result = compute_signal_characterization(dfs).set_index("Topic")
    assert result.loc[1, IMPACT_COLUMN] > result.loc[0, IMPACT_COLUMN]


def test_more_evidence_means_lower_uncertainty():
    # Same popularity; topic 1 has far more documents and sources -> lower uncertainty
    dfs = {WEAK_SIGNALS: _df([0, 1], [10, 10], [1, 100], [1, 20])}
    result = compute_signal_characterization(dfs).set_index("Topic")
    assert result.loc[1, UNCERTAINTY_COLUMN] < result.loc[0, UNCERTAINTY_COLUMN]


def test_signal_type_labelled():
    dfs = {
        WEAK_SIGNALS: _df([0], [10], [5], [2]),
        STRONG_SIGNALS: _df([1], [20], [8], [3]),
    }
    result = compute_signal_characterization(dfs).set_index("Topic")
    assert result.loc[0, "signal_type"] == WEAK_SIGNALS
    assert result.loc[1, "signal_type"] == STRONG_SIGNALS


def test_constant_metric_maps_to_midpoint():
    # All popularity equal -> impact 0.5 for every topic
    dfs = {WEAK_SIGNALS: _df([0, 1, 2], [7, 7, 7], [1, 2, 3], [1, 2, 3])}
    result = compute_signal_characterization(dfs)
    assert (result[IMPACT_COLUMN] == 0.5).all()


def test_missing_source_diversity_column_is_handled():
    df = pd.DataFrame(
        {"Topic": [0, 1], "Latest_Popularity": [1, 2], "Docs_Count": [3, 4]}
    )
    result = compute_signal_characterization({WEAK_SIGNALS: df})
    assert len(result) == 2
    assert result[UNCERTAINTY_COLUMN].between(0, 1).all()


def test_plot_returns_figure_with_traces():
    dfs = {
        WEAK_SIGNALS: _df([0, 1], [5, 20], [2, 8], [1, 4], titles=["A", "B"]),
        STRONG_SIGNALS: _df([2], [40], [30], [10], titles=["C"]),
    }
    char_df = compute_signal_characterization(dfs)
    fig = plot_signal_characterization(char_df)
    # One trace per signal type present
    assert len(fig.data) == 2


def test_plot_empty_is_empty_figure():
    fig = plot_signal_characterization(pd.DataFrame())
    assert len(fig.data) == 0
