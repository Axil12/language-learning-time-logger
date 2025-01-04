from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import streamlit as st
from streamlit_theme import st_theme
import pandas as pd
import july
import matplotlib.pyplot as plt
import plotly.express as px

from utils import (
    build_df_for_cumul_stack_plot,
    generate_july_plot,
    generate_calplot,
)

CSV_LOG_FILE = "time_logged.csv"
ACTIVITIES = [
    None,
    "anki",
    "anime",
    "youtube",
    "twitch",
    "reading",
    "textbook study",
]
TAGS = [
    None,
    "vocabulary",
    "grammar",
    "active immersion",
    "passive immersion",
]

colordict = {f: px.colors.qualitative.Prism[i] for i, f in enumerate(ACTIVITIES)}
# colordict = {f:  px.colors.sample_colorscale('Twilight', len(ACTIVITIES))[i] for i, f in enumerate(ACTIVITIES)}

st.set_page_config(page_title="JP Language Logger", page_icon="🇯🇵", layout="wide")
theme = st_theme()

if Path(CSV_LOG_FILE).exists():
    df_source = pd.read_csv(CSV_LOG_FILE, sep=";")
    df_source["date"] = pd.to_datetime(df_source["date"])
else:
    df_source = pd.DataFrame(columns=["date", "activity", "time", "tag", "comment"])
    df_source.to_csv(CSV_LOG_FILE, sep=";", index=False)

with st.sidebar:
    with st.form(key="add_new_entry_form"):
        st.markdown("# New entry")
        add_log_container = st.columns(6)
        time_mins = st.number_input("Time (minutes)", min_value=0, step=1)
        activity = st.radio("Activity", options=ACTIVITIES)
        tag = st.radio("Tag", options=TAGS)
        datetime_cols = st.columns(2)
        date = datetime_cols[0].date_input(label="Date")
        time = datetime_cols[1].time_input(
            "Time",
            datetime(1970, 1, 1, 0, 0),
            step=60,
            help="Leave at 0:00 if you want the current time.",
        )
        comment = st.text_input("Comment")
        add_log_button = st.form_submit_button(
            "Add log", icon="➕", use_container_width=True
        )
        message_container = st.empty()

if add_log_button:
    if activity is None:
        message_container.error("No activity was chosen !")
    else:
        if tag is None:
            tag = "none"
        if time == datetime(1970, 1, 1, 0, 0).time():
            time = datetime.now().time()
        new_log = pd.DataFrame(
            {
                "date": [datetime.combine(date, time)],
                "activity": [activity],
                "time": [time_mins],
                "tag": [tag],
                "comment": [comment],
            }
        )
        df_source = pd.concat([df_source, new_log], ignore_index=True)

        df_source.to_csv(CSV_LOG_FILE, sep=";", index=False)
        st.rerun()

df = df_source.copy()

###############################################################################
#
# Filters dropdown
#
###############################################################################

with st.expander("Filters"):
    all_activities = set(ACTIVITIES + list(df["activity"].unique()))
    all_activities = [x for x in all_activities if x is not None]
    all_tags = set(TAGS + list(df["tag"].unique()))
    all_tags = [x for x in all_tags if x is not None]
    filters_container = st.container()
    filtered_in_container_cols = filters_container.columns(2)
    filtered_in_container_cols[0].multiselect(
        "Filter in activities", all_activities, key="filtered_in_activities"
    )
    filtered_in_container_cols[1].multiselect(
        "Filter in tags", all_tags, key="filtered_in_tags"
    )
    filtered_out_container_cols = filters_container.columns(2)
    filtered_out_container_cols[0].multiselect(
        "Filter out activities", all_activities, key="filtered_out_activities"
    )
    filtered_out_container_cols[1].multiselect(
        "Filter out tags", all_tags, key="filtered_out_tags"
    )

    df = df[
        df["activity"].isin(
            st.session_state.get("filtered_in_activities") or all_activities
        )
    ]
    df = df[df["tag"].isin(st.session_state.get("filtered_in_tags") or all_tags)]
    df = df[~df["activity"].isin(st.session_state.get("filtered_out_activities", []))]
    df = df[~df["tag"].isin(st.session_state.get("filtered_out_tags", []))]
    if not len(df):  # If the dataframe is empty
        st.error("The dataframe is empty of logs !")
        st.stop()

###############################################################################
#
# Dataframe container
#
###############################################################################

df_container = st.container()
df_container_cols = df_container.columns([0.7, 0.3])

# Displays dataframe
df_container_cols[0].dataframe(df.sort_values(by="date", ascending=False), width=5000)

# Pie chart that displays the distribution of activities
min_time = datetime.combine(datetime.now().date(), datetime.min.time()) - timedelta(days=7)
df_7days = df[df["date"] > min_time].copy()
min_time = datetime.combine(datetime.now().date(), datetime.min.time()) - timedelta(days=30)
df_30days = df[df["date"] > min_time].copy()
df_container_cols[1].markdown(
    f"""
    Hours logged : **{sum(df["time"])/60:.2f} h**\n
    Average per day : \n
    All time : **{sum(df["time"])/60/((max(df["date"]) - min(df["date"])).days+1):.2f} h/d** | 
    This month : **{sum(df_30days["time"])/60/((max(df["date"]) - min(df_30days["date"])).days+1):.2f} h/d** | 
    This week : **{sum(df_7days["time"])/60/((max(df_7days["date"]) - min(df_7days["date"])).days+1):.2f} h/d**\n
    """
)
df_tmp = df.copy()
df_tmp["total_time"] = df["time"].sum() / 60  # Conversion from minutes to hours
df_tmp["time"] = df_tmp["time"] / 60  # Conversion from minutes to hours
fig = px.sunburst(
    df_tmp,
    path=["total_time", "activity", "tag"],
    values="time",
)
fig.update_layout(margin=dict(t=0, b=0, r=10, l=10), height=260)
df_container_cols[1].plotly_chart(fig, key="Activities pie chart")

# Calendar heatmap
df_container.plotly_chart(
    generate_calplot(df, dark_mode=theme.get("base") == "dark" if theme else None)
)

###############################################################################
#
# Stats : Stats over time
#
###############################################################################

st.markdown("## Stats")
stats_container = st.container()

stats_container.markdown("### Activities")
stacked_plots = stats_container.columns(3)
stacked_plots[0].markdown("#### This week")
min_time = datetime.combine(datetime.now().date(), datetime.min.time()) - timedelta(days=7)
df_tmp = df[df["date"] > min_time].copy()
df_tmp["day"] = df_tmp["date"].dt.date
df_tmp = df_tmp.groupby(["day", "activity"])["time"].sum().reset_index()
df_tmp["time_hours"] = df_tmp["time"] / 60
df_tmp = df_tmp.sort_values(by="activity")
fig = px.area(
    df_tmp,
    x="day",
    y="time_hours",
    color="activity",
    line_shape="spline",
    color_discrete_map=colordict,
)
fig.update_layout(
    legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    margin=dict(t=20, b=0, r=0, l=0),
    height=300,
)
stacked_plots[0].plotly_chart(fig, key="Activity time hours 7day")
stacked_plots[1].markdown("#### This month")
min_time = datetime.combine(datetime.now().date(), datetime.min.time()) - timedelta(days=30)
df_tmp = df[df["date"] > min_time].copy()
df_tmp["day"] = df_tmp["date"].dt.date
df_tmp = df_tmp.groupby(["day", "activity"])["time"].sum().reset_index()
df_tmp["time_hours"] = df_tmp["time"] / 60
df_tmp = df_tmp.sort_values(by="activity")
fig = px.area(
    df_tmp,
    x="day",
    y="time_hours",
    color="activity",
    line_shape="spline",
    color_discrete_map=colordict,
)
fig.update_layout(
    legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    margin=dict(t=20, b=0, r=0, l=0),
    height=300,
)
stacked_plots[1].plotly_chart(fig, key="Activity time hours 30day")
stacked_plots[2].markdown("#### This year")
min_time = datetime.combine(datetime.now().date(), datetime.min.time()) - timedelta(days=365)
df_tmp = df[df["date"] > min_time].copy()
df_tmp["day"] = df_tmp["date"].dt.date
df_tmp = df_tmp.groupby(["day", "activity"])["time"].sum().reset_index()
df_tmp["time_hours"] = df_tmp["time"] / 60
df_tmp = df_tmp.sort_values(by="activity")
fig = px.area(
    df_tmp,
    x="day",
    y="time_hours",
    color="activity",
    line_shape="spline",
    color_discrete_map=colordict,
)
fig.update_layout(
    legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    margin=dict(t=20, b=0, r=0, l=0),
    height=300,
)
stacked_plots[2].plotly_chart(fig, key="Activity time hours 365day")

stats_container.markdown("### Cumulative time")
df_tmp["Cumulative time"] = df_tmp["time"].cumsum()
cumul_stacked_plots = stats_container.columns(3)
cumul_stacked_plots[0].markdown("#### This week")
min_time = datetime.combine(datetime.now().date(), datetime.min.time()) - timedelta(days=7)
df_tmp = df[df["date"] > min_time].copy()
df_tmp = build_df_for_cumul_stack_plot(df_tmp)
df_tmp["Cumulative time hours"] = df_tmp["Cumulative time"] / 60
df_tmp = df_tmp.sort_values(by="activity")
fig = px.area(
    df_tmp,
    x="day",
    y="Cumulative time hours",
    color="activity",
    line_shape="spline",
    color_discrete_map=colordict,
)
fig.update_layout(
    legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    margin=dict(t=20, b=0, r=0, l=0),
    height=300,
)
cumul_stacked_plots[0].plotly_chart(fig, key="Cumulative time hours 7day")
cumul_stacked_plots[1].markdown("#### This month")
min_time = datetime.combine(datetime.now().date(), datetime.min.time()) - timedelta(days=30)
df_tmp = df[df["date"] > min_time].copy()
df_tmp = build_df_for_cumul_stack_plot(df_tmp)
df_tmp["Cumulative time hours"] = df_tmp["Cumulative time"] / 60
df_tmp = df_tmp.sort_values(by="activity")
fig = px.area(
    df_tmp,
    x="day",
    y="Cumulative time hours",
    color="activity",
    line_shape="spline",
    color_discrete_map=colordict,
)
fig.update_layout(
    legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    margin=dict(t=20, b=0, r=0, l=0),
    height=300,
)
cumul_stacked_plots[1].plotly_chart(fig, key="Cumulative time hours 30day")
cumul_stacked_plots[2].markdown("#### This year")
min_time = datetime.combine(datetime.now().date(), datetime.min.time()) - timedelta(days=365)
df_tmp = df[df["date"] > min_time].copy()
df_tmp = build_df_for_cumul_stack_plot(df_tmp)
df_tmp["Cumulative time hours"] = df_tmp["Cumulative time"] / 60
df_tmp = df_tmp.sort_values(by="activity")
fig = px.area(
    df_tmp,
    x="day",
    y="Cumulative time hours",
    color="activity",
    line_shape="spline",
    color_discrete_map=colordict,
)
fig.update_layout(
    legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    margin=dict(t=20, b=0, r=0, l=0),
    height=300,
)
cumul_stacked_plots[2].plotly_chart(fig, key="Cumulative time hours 365day")


###############################################################################
#
# Histogram : repartition of the times at which I finish each activity
#
###############################################################################

stats_container.markdown("### Time repartition")
stats_container.markdown("This describes the times at which activities tend to finish.")
# First, we get rid of the data points that have a recorded hour at midnight. As those are considered to have an unknown time.
df_tmp = df[df["date"].dt.time != datetime(1970, 1, 1, 0, 0).time()].copy()
df_tmp["time"] = df_tmp["date"].dt.time
df_tmp["hour"] = df_tmp["date"].dt.hour
counts, bins = np.histogram(df_tmp["hour"], bins=range(0, 25, 1))
bins = (
    0.5 * (bins[:-1] + bins[1:]) - 0.5
)  # I think this handles the position of the xticks ?
fig = px.bar(
    x=bins,
    y=counts,
    labels={"x": "hour", "y": "count"},
    color=bins,
    color_continuous_scale=px.colors.sequential.RdPu,
)
fig.update_layout(
    bargap=0.02,
    barcornerradius=3,
    coloraxis_showscale=False,
    xaxis=dict(
        tickmode="array",
        tickvals=[i for i in range(24)],
        ticktext=[f"{i}:00" for i in range(24)],
        tickangle=40,
    ),
)
stats_container.plotly_chart(fig, key="histogram_activity_times")
