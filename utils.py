from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import july
from plotly_calplot import calplot
import plotly.express as px


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


def generate_calplot(_df: pd.DataFrame, dark_mode: bool = False):
    df = _df.copy()
    df["day"] = df["date"].dt.date
    df = df.groupby(["day"])["time"].sum().reset_index()
    df = df.fillna(0)
    fig = calplot(
        df,
        x="day",
        y="time",
        cmap_min=0,
        # colorscale="magenta",
        colorscale="rdpu",
        gap=1,
        showscale=True,
        month_lines_width=3,
        month_lines_color="#0E1117" if dark_mode else "#FFFFFF",
    )
    fig.update_layout(plot_bgcolor="#0E1117" if dark_mode else "#FFFFFF")
    return fig




####

from typing import Any, List, Tuple
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd


def fill_empty_with_zeros(
    selected_year_data: pd.DataFrame,
    x: str,
    year: int,
    start_month: int = 1,
    end_month: int = 12,
) -> pd.DataFrame:
    """
    Taken from https://github.com/brunorosilva/plotly-calplot/blob/main/plotly_calplot/utils.py

    Fills empty dates with zeros in the selected year data.

    Args:
        selected_year_data (DataFrame): The data for the selected year.
        x (str): The column name for the date values.
        year (int): The year for which the data is being filled.
        start_month (int): The starting month of the year.
        end_month (int): The ending month of the year.

    Returns:
        pd.DataFrame: The final DataFrame with empty dates filled with zeros.
    """
    if end_month != 12:
        last_date = datetime(year, end_month + 1, 1) + timedelta(days=-1)
    else:
        last_date = datetime(year, 1, 1) + timedelta(days=-1)
    year_min_date = date(year=year, month=start_month, day=1)
    year_max_date = date(year=year, month=end_month, day=last_date.day)
    df = pd.DataFrame({x: pd.date_range(year_min_date, year_max_date)})
    final_df = df.merge(selected_year_data, how="left")
    return final_df


def get_date_coordinates(
    data: pd.DataFrame, date_col_name: str
) -> Tuple[Any, List[float], List[int]]:
    """
    Taken from https://github.com/brunorosilva/plotly-calplot/blob/main/plotly_calplot/date_extractors.py
    """
    month_days = []
    for m in data[date_col_name].dt.month.unique():
        month_days.append(
            data.loc[data[date_col_name].dt.month == m, date_col_name].max().day
        )

    month_positions = np.linspace(1.5, 50, 12)
    weekdays_in_year = [i.weekday() for i in data[date_col_name]]

    # sometimes the last week of the current year conflicts with next year's january
    # pandas uses ISO weeks, which will give those weeks the number 52 or 53, but this
    # is bad news for this plot therefore we need a correction to use Gregorian weeks,
    # for a more in-depth explanation check
    # https://stackoverflow.com/questions/44372048/python-pandas-timestamp-week-returns-52-for-first-day-of-year
    weeknumber_of_dates = data[date_col_name].dt.strftime("%W").astype(int).tolist()

    return month_positions, weekdays_in_year, weeknumber_of_dates
