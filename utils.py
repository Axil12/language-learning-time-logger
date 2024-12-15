import pandas as pd
import matplotlib.pyplot as plt
import july
from plotly_calplot import calplot
import plotly.express as px


def build_df_for_cumul_stack_plot(_df: pd.DataFrame) -> pd.DataFrame:
    df = _df.copy()
    df["day"] = df["date"].dt.date
    df = df.groupby(["day", "activity"])["time"].sum().reset_index()
    all_days = pd.date_range(start=df["day"].min(), end=df["day"].max(), freq="D").date
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
