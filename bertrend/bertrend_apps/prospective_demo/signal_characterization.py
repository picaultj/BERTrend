#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

"""Characterize signals on an impact x uncertainty map for the prospective demo.

This is a heuristic, data-derived characterization built entirely from the
scalar metrics already stored in the signal dataframes (no LLM call and no
popularity time series required), so it works retroactively on existing models:

- impact: normalized ``Latest_Popularity`` (magnitude of attention)
- uncertainty: ``1 - normalized(evidence)`` where evidence blends ``Docs_Count``
  and ``Source_Diversity`` — a thinly-supported topic (few documents, few
  sources) is treated as more uncertain.

Both axes are min-max normalized to ``[0, 1]`` across the combined set of
signals for the selected model/timestamp.
"""

import pandas as pd
import plotly.graph_objects as go

from bertrend.bertrend_apps.prospective_demo import (
    LLM_TOPIC_TITLE_COLUMN,
    NOISE,
    STRONG_SIGNALS,
    WEAK_SIGNALS,
)

IMPACT_COLUMN = "impact"
UNCERTAINTY_COLUMN = "uncertainty"

# Metric columns used to derive the axes.
_POPULARITY_COLUMN = "Latest_Popularity"
_DOCS_COLUMN = "Docs_Count"
_DIVERSITY_COLUMN = "Source_Diversity"

_OUTPUT_COLUMNS = [
    "Topic",
    "title",
    "signal_type",
    _POPULARITY_COLUMN,
    _DOCS_COLUMN,
    _DIVERSITY_COLUMN,
    IMPACT_COLUMN,
    UNCERTAINTY_COLUMN,
]

# Plot colors per signal type.
SIGNAL_TYPE_COLORS = {
    WEAK_SIGNALS: "orange",
    STRONG_SIGNALS: "green",
    NOISE: "grey",
}


def _minmax(series: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]; a constant series maps to 0.5."""
    values = series.astype(float)
    low, high = values.min(), values.max()
    if pd.isna(low) or high == low:
        return pd.Series(0.5, index=series.index)
    return (values - low) / (high - low)


def compute_signal_characterization(
    dfs_topics: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine the weak/strong/noise topic dataframes and derive impact and
    uncertainty scores per topic.

    Parameters
    ----------
    dfs_topics : dict[str, pd.DataFrame]
        Mapping of signal type (NOISE / WEAK_SIGNALS / STRONG_SIGNALS) to its
        topic dataframe, as returned by ``get_df_topics``.

    Returns
    -------
    pd.DataFrame
        One row per topic with columns ``Topic``, ``title``, ``signal_type``,
        the raw metrics, and the normalized ``impact`` / ``uncertainty`` scores.
        Empty input yields an empty dataframe with the expected columns.
    """
    frames = []
    for signal_type in (WEAK_SIGNALS, STRONG_SIGNALS, NOISE):
        df = dfs_topics.get(signal_type) if dfs_topics else None
        if df is None or df.empty:
            continue
        sub = df.copy()
        sub["signal_type"] = signal_type
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)

    # Guarantee the metric columns exist (older/partial data).
    for col in (_POPULARITY_COLUMN, _DOCS_COLUMN, _DIVERSITY_COLUMN):
        if col not in combined.columns:
            combined[col] = 0.0

    impact = _minmax(combined[_POPULARITY_COLUMN])
    evidence = (
        _minmax(combined[_DOCS_COLUMN]) + _minmax(combined[_DIVERSITY_COLUMN])
    ) / 2.0

    combined[IMPACT_COLUMN] = impact
    combined[UNCERTAINTY_COLUMN] = 1.0 - evidence

    if LLM_TOPIC_TITLE_COLUMN in combined.columns:
        combined["title"] = combined[LLM_TOPIC_TITLE_COLUMN].fillna("").astype(str)
    else:
        combined["title"] = ""

    return combined[_OUTPUT_COLUMNS]


def plot_signal_characterization(
    char_df: pd.DataFrame,
    signal_type_labels: dict[str, str] | None = None,
    title: str | None = None,
    impact_label: str = "Impact",
    uncertainty_label: str = "Uncertainty",
) -> go.Figure:
    """Build an impact x uncertainty scatter (quadrant map) of the signals.

    One marker per topic: x=uncertainty, y=impact, colored by signal type and
    sized by document count. Dashed lines at 0.5 split the four quadrants.
    """
    fig = go.Figure()
    if char_df is None or char_df.empty:
        return fig

    labels = signal_type_labels or {}
    # Marker size scaled by document count across the whole set.
    marker_size = 10.0 + 20.0 * _minmax(char_df[_DOCS_COLUMN])
    plot_df = char_df.assign(_marker_size=marker_size)

    for signal_type, group in plot_df.groupby("signal_type"):
        fig.add_trace(
            go.Scatter(
                x=group[UNCERTAINTY_COLUMN],
                y=group[IMPACT_COLUMN],
                mode="markers",
                name=labels.get(signal_type, signal_type),
                marker=dict(
                    size=group["_marker_size"],
                    color=SIGNAL_TYPE_COLORS.get(signal_type, "blue"),
                    line=dict(width=1, color="white"),
                    opacity=0.8,
                ),
                text=group["title"],
                customdata=group[
                    ["Topic", _DOCS_COLUMN, _DIVERSITY_COLUMN, _POPULARITY_COLUMN]
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"{uncertainty_label}: %{{x:.2f}}<br>"
                    f"{impact_label}: %{{y:.2f}}<br>"
                    "Topic: %{customdata[0]}<br>"
                    "Docs: %{customdata[1]}<br>"
                    "Sources: %{customdata[2]}<br>"
                    "Popularity: %{customdata[3]:.2f}<extra></extra>"
                ),
            )
        )

    # Quadrant separators.
    fig.add_hline(y=0.5, line_dash="dash", line_color="lightgrey")
    fig.add_vline(x=0.5, line_dash="dash", line_color="lightgrey")

    fig.update_layout(
        title=title or "Signal map (impact vs uncertainty)",
        xaxis_title=uncertainty_label,
        yaxis_title=impact_label,
        xaxis=dict(range=[-0.05, 1.05]),
        yaxis=dict(range=[-0.05, 1.05]),
        legend_title="",
        hovermode="closest",
    )
    return fig
