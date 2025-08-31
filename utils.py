from typing import Any, List, Tuple, Union
from datetime import datetime, timedelta, date

import pandas as pd
import matplotlib.pyplot as plt
import july
from plotly_calplot import calplot
import plotly.express as px
import numpy as np


def threshold_color_scale(
    threshold: float, min_val: float, max_val: float, soft_threshold: float = 0.0
) -> list[tuple[float, str]]:
    thresh = (threshold - min_val) / (max_val - min_val)

    cmap_1 = [
        "rgb(255,247,243)",
        "rgb(253,224,221)",
        "rgb(252,197,192)",
        "rgb(250,159,181)",
        "rgb(247,104,161)",
        "rgb(221,52,151)",
    ]  # first part of px.colors.sequential.RdPu
    cmap_2 = px.colors.sequential.Blugrn[1:]

    if soft_threshold > 0:
        cmap_1 = zip(
            list(
                np.linspace(
                    start=0, stop=thresh * (1 - soft_threshold), num=len(cmap_1)
                )
            ),
            cmap_1,
        )
        cmap_2 = zip(
            list(
                np.linspace(
                    start=min(1, thresh * (1 + soft_threshold)), stop=1, num=len(cmap_2)
                )
            ),
            cmap_2,
        )
    else:
        cmap_1 = zip(list(np.linspace(start=0, stop=thresh, num=len(cmap_1))), cmap_1)
        cmap_2 = zip(list(np.linspace(start=thresh, stop=1, num=len(cmap_2))), cmap_2)

    return list(cmap_1) + list(cmap_2)


def build_df_for_cumul_stack_plot(_df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    df = _df.copy()
    df["day"] = df["date"].dt.date
    df = df.groupby(["day", "activity"])["time"].sum().reset_index()
    # all_days = pd.date_range(start=df["day"].min(), end=df["day"].max(), freq="D").date
    all_days = df["day"].unique()
    if len(all_days) > 0:
        today_dt = datetime.combine(datetime.now().date(), datetime.min.time())
        start = today_dt - timedelta(days=n_days)
        start = max(start, pd.to_datetime(df["day"]).min())
        end = today_dt
        all_days = pd.date_range(start=start, end=end, freq="D").date
    all_activities = df["activity"].unique()
    index = pd.MultiIndex.from_product(
        [all_days, all_activities], names=["day", "activity"]
    )
    df = df.set_index(["day", "activity"]).reindex(index, fill_value=0).reset_index()
    df["Cumulative time"] = df.groupby(["activity"])["time"].cumsum()
    return df


def generate_july_plot(_df: pd.DataFrame, dark_mode: bool = False) -> plt.Figure:
    df = _df.copy()
    df["day"] = df["date"].dt.date
    df = df.groupby(["day"])["time"].sum().reset_index()

    tmp = pd.date_range(
        start=df["day"].max() - pd.Timedelta(days=365),
        end=df["day"].min() - pd.Timedelta(days=1),
        freq="D",
    ).date
    tmp = pd.DataFrame({"day": tmp, "time": [0] * len(tmp)})
    df = pd.concat([tmp, df])
    fig, ax = plt.subplots(
        facecolor="#0E1117" if dark_mode else "#FFFFFF",
    )
    july.heatmap(
        dates=df["day"],
        data=df["time"],
        ax=ax,
        cmin=0,
        colorbar=True,
        cmap="RdPu",
        dpi=100,
        month_grid=True,
        horizontal=True,
        fontsize=4,
        value_label=False,
        date_label=False,
        frame_on=False,
    )
    ax.tick_params(colors="#FFFFFF" if dark_mode else "#000000")
    ax.collections[-1].colorbar.ax.tick_params(
        colors="#FFFFFF" if dark_mode else "#000000"
    )
    return fig


def generate_calplot(
    _df: pd.DataFrame,
    dark_mode: bool = False,
    cmap_min: Union[float, None] = None,
    cmap_max: Union[float, None] = None,
    cmap_threshold: Union[float, None] = None,
):
    if cmap_min is None:
        cmap_min = 0
    if cmap_min is not None and cmap_max is not None and cmap_threshold is not None:
        colorscale = threshold_color_scale(
            cmap_threshold, cmap_min, cmap_max, soft_threshold=0
        )
    else:
        colorscale = px.colors.sequential.RdPu

    df = _df.copy()
    df["day"] = df["date"].dt.date
    df = df.groupby(["day"])["time"].sum().reset_index()
    df = df.fillna(0)
    fig = calplot(
        df,
        x="day",
        y="time",
        cmap_min=cmap_min,
        cmap_max=cmap_max,
        # colorscale="magenta",
        # colorscale="rdpu",
        # colorscale=px.colors.sequential.RdPu,
        colorscale=colorscale,
        gap=1,
        showscale=True,
        month_lines_width=3,
        month_lines_color="#0E1117" if dark_mode else "#FFFFFF",
    )
    fig.update_layout(plot_bgcolor="#0E1117" if dark_mode else "#FFFFFF")
    return fig


def group_days_for_plotting(df_source, n_days_group=7, period_days=365):
    """Group daily activity data into N-day periods for plotting.
    
    Args:
        df_source (pd.DataFrame): Original dataframe with columns 'day', 'activity', and 'time_hours'.
        n_days_group (int): Number of days per group.
        period_days (int): Number of days to include in the data before grouping.

    Returns:
        pd.DataFrame: Aggregated dataframe with columns 'start_date', 'activity', 'time_hours', and 'time_str'.
    """
    # Ensure 'date' is datetime and create 'day' column
    df_source = df_source.copy()
    df_source["date"] = pd.to_datetime(df_source["date"], errors="coerce")
    df_source["day"] = pd.to_datetime(df_source["date"].dt.date)

    # Convert time to hours
    df_source["time_hours"] = df_source["time"] / 60

    # Filter data for the last `period_days`
    cutoff_date = pd.Timestamp.today().normalize() - timedelta(days=period_days)
    df_to_plot = df_source[df_source["day"] > cutoff_date].copy()

    # Assign a group number for each N-day block
    df_to_plot["day_num"] = (df_to_plot["day"] - df_to_plot["day"].min()).dt.days
    df_to_plot["group"] = df_to_plot["day_num"].floordiv(n_days_group)

    # Get the start date for each group (to use as x-axis labels)
    group_start_dates = (
        df_to_plot.groupby("group")["day"].min().reset_index().rename(columns={"day": "start_date"})
    )
    df_to_plot = df_to_plot.merge(group_start_dates, on="group", how="left")

    # Aggregate time per activity per group
    df_grouped = df_to_plot.groupby(["start_date", "activity"], as_index=False)["time_hours"].sum()
    df_grouped["time_str"] = (
        (df_grouped["time_hours"].astype(int)).astype(str)
        + "h"
        + ((df_grouped["time_hours"] % 1) * 60).astype(int).map("{:02d}".format)
    )

    return df_grouped

