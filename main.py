from typing import Any, List, Tuple
from random import randrange, randint
import datetime

# from datetime.datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import numpy as np

from utils import get_date_coordinates, fill_empty_with_zeros


def random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + datetime.timedelta(seconds=random_second)


def towers(a, e, pos_x, pos_y, min_value, max_value, colorscale="rdpu"):
    # create points
    x, y, z = np.meshgrid(
        np.linspace(pos_x - a / 2, pos_x + a / 2, 2),
        np.linspace(pos_y - a / 2, pos_y + a / 2, 2),
        np.linspace(0, e, 2),
    )
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    # color = "ff0000"
    color = sample_colorscale("rdpu", [e / max_value])[
        0
    ]  # , low=min_value, high=max_value)[0]

    tower = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        alphahull=1,
        flatshading=True,
        color=color,
    )
    return tower


def create_month_lines(
    cplt: List[go.Figure],
    month_lines_color: str,
    month_lines_width: int,
    data: pd.DataFrame,
    weekdays_in_year: List[float],
    weeknumber_of_dates: List[int],
) -> go.Figure:
    kwargs = dict(
        mode="lines",
        line=dict(color=month_lines_color, width=month_lines_width),
        hoverinfo="skip",
    )
    for date, dow, wkn in zip(data, weekdays_in_year, weeknumber_of_dates):
        if date.day == 1:
            cplt += [go.Scatter(x=[wkn - 0.5, wkn - 0.5], y=[dow - 0.5, 6.5], **kwargs)]
            if dow:
                cplt += [
                    go.Scatter(
                        x=[wkn - 0.5, wkn + 0.5], y=[dow - 0.5, dow - 0.5], **kwargs
                    ),
                    go.Scatter(x=[wkn + 0.5, wkn + 0.5], y=[dow - 0.5, -0.5], **kwargs),
                ]
    return cplt


def get_month_names(
    data: pd.DataFrame, x: str, start_month: int = 1, end_month: int = 12
) -> List[str]:
    start_month_names_filler = [None] * (start_month - 1)
    end_month_names_filler = [None] * (12 - end_month)
    month_names = list(
        start_month_names_filler
        + data[x].dt.month_name().unique().tolist()
        + end_month_names_filler
    )
    return month_names


def main():
    n = 200
    date_min = datetime.datetime.strptime("1/4/2024 00:00", "%d/%m/%Y %H:%M")
    data = {
        "date": [date_min + datetime.timedelta(days=i) for i in range(n)],
        "val": [randint(1, 5) for _ in range(n)],
    }
    df = pd.DataFrame(data)
    df = fill_empty_with_zeros(df.loc[df["date"].dt.year == 2024], "date", 2024).fillna(
        0
    )
    month_names = get_month_names(df, "date")
    month_positions, weekdays_in_year, weeknumber_of_dates = get_date_coordinates(
        df, "date"
    )
    month_shift = 1.5
    month_positions = [m + (i + 1) * month_shift for i, m in enumerate(month_positions)]

    colorscale = "rdpu"
    min_val, max_val = 0, max(df["val"])

    fig = go.Figure(
        layout={"scene": {"aspectmode": "data"}, "height": 700, "width": 800}
    )
    for m, x, y, z, date in zip(
        df["date"].dt.month,
        weeknumber_of_dates,
        weekdays_in_year,
        df["val"],
        df["date"],
    ):
        is_month_beginning = False
        is_month_end = False
        if date.day <= 7:
            is_month_beginning = True
        elif (24 <= date.day <= 31) and date.month % 2 == 1:
            is_month_end = True
        elif (23 <= date.day <= 30) and date.month % 2 == 0:
            is_month_end = True

        fig.add_trace(
            towers(
                1, z, x + m * month_shift, y, min_val, max_val, colorscale=colorscale
            )
        )

    layout = go.Layout(
        title="calplot",
        scene=dict(
            xaxis_title="Month",
            yaxis_title="Day",
            zaxis_title="Value",
            yaxis=dict(
                showline=False,
                showgrid=False,
                zeroline=False,
                tickmode="array",
                ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                tickvals=[0, 1, 2, 3, 4, 5, 6],
                autorange="reversed",
            ),
            xaxis=dict(
                showline=False,
                showgrid=False,
                zeroline=False,
                tickmode="array",
                ticktext=month_names,
                tickvals=month_positions,
                autorange="reversed",
            ),
        ),
    )
    fig.update_layout(layout)
    fig.update_xaxes(layout["xaxis"])
    fig.update_yaxes(layout["yaxis"])
    fig.show()


if __name__ == "__main__":
    main()
