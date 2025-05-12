from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
from streamlit_theme import st_theme
import pandas as pd
import plotly.express as px

from utils import (
    build_df_for_cumul_stack_plot,
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

# We get today's date at midnight. This will be usefull in multiple places in the page to compute time slices
today_dt = datetime.combine(datetime.now().date(), datetime.min.time())

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
    elif time_mins <= 0:
        message_container.error("The time spent can't be 0.")
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
df_7days = df[df["date"] > today_dt - timedelta(days=6)].copy()
df_30days = df[df["date"] > today_dt - timedelta(days=30)].copy()
avg_7_days = sum(df_7days["time"]) / 7
avg_30_days = sum(df_30days["time"]) / 30
avg_all_time = sum(df["time"]) / ((today_dt - min(df["date"])).days + 1)

df_container_cols[1].markdown(
    f"""
    Hours logged : **{sum(df["time"])/60:.2f} h**\n
    ###### Average per day : \n
    This week : **{int(avg_7_days//60)}h{int(avg_7_days%60):02d}** | 
    This month : **{int(avg_30_days//60)}h{int(avg_30_days%60):02d}** | 
    All time : **{int(avg_all_time//60)}h{int(avg_all_time%60):02d}** 
    """
)
df_tmp = df.copy()
df_tmp["total_time"] = df["time"].sum() / 60  # Conversion from minutes to hours
df_tmp["time"] = df_tmp["time"] / 60  # Conversion from minutes to hours
fig = px.sunburst(
    df_tmp,
    path=["total_time", "activity", "tag"],
    values="time",
    labels={"time": "Time"},
    hover_data=dict(time=":.2f", activity=True, tag=True),
)
# ['date', 'activity', 'time', 'tag', 'comment', 'total_time']
fig.update_layout(margin=dict(t=0, b=0, r=10, l=10), height=260)
# Directly modifying the HTML to clean it because I haven't found any other way to do it
fig.data[0].hovertemplate = fig.data[0].hovertemplate.replace(r"id=%{id}<br>", r"")
fig.data[0].hovertemplate = fig.data[0].hovertemplate.replace(
    r"parent=%{parent}<br>", r""
)
fig.data[0].customdata[-1][0] = round(fig.data[0].customdata[-1][0], 2)
val = float(fig.data[0].labels[-1])
fig.data[0].labels[-1] = f"{int(val)}h {int((val-int(val))*60):02d}m"
df_container_cols[1].plotly_chart(fig, key="Activities pie chart")

###############################################################################
#
# Calendar heatmap
#
###############################################################################

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

### Time spent each day
stats_container.markdown("### Activities")
df_tmp = df.copy()
stacked_bar_plots = stats_container.columns(3)
for i, (tag, n_days) in enumerate(
    {"This week": 6, "This month": 30, "This year": 365}.items()
):
    stacked_bar_plots[i].markdown(f"#### {tag}")
    
    df_tmp = df[df["date"] > today_dt - timedelta(days=n_days)].copy()
    if len(df_tmp) == 0:
        stacked_bar_plots[i].markdown(f"No data")
        continue
    df_tmp["day"] = df_tmp["date"].dt.date
    day_first_entry = min(df_tmp["day"])
    all_days_in_range = pd.DataFrame(
        {"day": [(today_dt - timedelta(days=i)).date() for i in range(n_days)]}
    )
    df_tmp = all_days_in_range.merge(df_tmp, on="day", how="left")
    df_tmp["activity"] = df_tmp["activity"].fillna("")
    df_tmp["time"] = df_tmp["time"].fillna(0)
    df_tmp = df_tmp[df_tmp["day"] >= day_first_entry]
    df_tmp = df_tmp.groupby(["day", "activity"])["time"].sum().reset_index()
    df_tmp["time_hours"] = df_tmp["time"] / 60
    df_tmp["time_str"] = (
        (df_tmp["time"] // 60).astype(int).astype(str)
        + "h"
        + (df_tmp["time"] % 60).astype(int).map("{:02d}".format)
    )
    df_tmp = df_tmp.sort_values(by="activity")
    fig = px.bar(
        df_tmp,
        x="day",
        y="time_hours",
        color="activity",
        color_discrete_map=colordict,
        opacity=0.6,
        labels={"activity": "Activity", "day": "Day", "time_str": "Time"},
        hover_data=dict(time_hours=False, time_str=True),
    )
    fig.update_layout(
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        margin=dict(t=20, b=0, r=0, l=0),
        height=300,
        bargap=0.01,
        barcornerradius=4 - i,
    )

    stacked_bar_plots[i].plotly_chart(fig, key=f"Activity time hours {n_days}day")

### Cumulative time spent
stats_container.markdown("### Cumulative time")
df_tmp = df.copy()
df_tmp["Cumulative time"] = df_tmp["time"].cumsum()
cumul_stacked_plots = stats_container.columns(3)
for i, (tag, n_days) in enumerate(
    {"This week": 6, "This month": 30, "This year": 365}.items()
):
    df_tmp = df[df["date"] > today_dt - timedelta(days=n_days)].copy()
    df_tmp = build_df_for_cumul_stack_plot(df_tmp, n_days)
    df_tmp["Cumulative time hours"] = df_tmp["Cumulative time"] / 60
    df_tmp["time_str"] = (
        (df_tmp["Cumulative time"] // 60).astype(int).astype(str)
        + "h"
        + (df_tmp["Cumulative time"] % 60).astype(int).map("{:02d}".format)
    )
    df_tmp = df_tmp.sort_values(by="activity")
    fig = px.area(
        df_tmp,
        x="day",
        y="Cumulative time hours",
        color="activity",
        line_shape="spline",
        color_discrete_map=colordict,
        labels={"activity": "Activity", "day": "Day", "time_str": "Time"},
        hover_data={"Cumulative time hours": False, "time_str": True},
    )
    fig.update_layout(
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        margin=dict(t=20, b=0, r=0, l=0),
        height=300,
    )
    cumul_stacked_plots[i].markdown(f"#### {tag}")
    cumul_stacked_plots[i].plotly_chart(fig, key=f"Cumulative time hours {n_days}day")


###############################################################################
#
# Histogram : repartition of the times at which I finish each activity
#
###############################################################################
stats_container.markdown("### Time repartition")
hist_cols = stats_container.columns(2)
hist_cols[0].markdown("#### Activity counts")
# First, we get rid of the data points that have a recorded hour at midnight. As those are considered to have an unknown time.
df_tmp = df[df["date"].dt.time != datetime(1970, 1, 1, 0, 0).time()].copy()
df_tmp["hour"] = df_tmp["date"].dt.hour
df_grouped = df_tmp.groupby(["hour", "activity"]).size().reset_index(name="count")
# Pivot the data to have activity types as columns
pivot_table = df_grouped.pivot(index="hour", columns="activity", values="count").fillna(
    0
)
full_hours = pd.DataFrame({"hour": range(24)})  # Add missing hours (0 to 23)
pivot_table = full_hours.merge(pivot_table, on="hour", how="left").fillna(0)
fig = px.bar(
    pivot_table,
    x="hour",
    y=pivot_table.columns[1:],
    labels={"value": "Count", "hour": "Hour of the Day"},
    barmode="stack",
    color_discrete_map=colordict,
    opacity=0.6,
)
fig.update_layout(
    bargap=0.05,
    barcornerradius=3,
    coloraxis_showscale=False,
    xaxis=dict(
        tickmode="array",
        tickvals=[i for i in range(24)],
        ticktext=[f"{i}:00" for i in range(24)],
        tickangle=40,
    ),
    legend=dict(x=0, y=1, title="Activity"),
)
hist_cols[0].plotly_chart(fig, key="histogram_activity_times")

hist_cols[1].markdown("#### Cumulated time")
# First, we get rid of the data points that have a recorded hour at midnight. As those are considered to have an unknown time.
df_tmp = df[df["date"].dt.time != datetime(1970, 1, 1, 0, 0).time()].copy()
df_tmp["hour"] = df_tmp["date"].dt.hour
df_tmp["time"] = df_tmp["time"] / 60
df_grouped = df_tmp.groupby(["hour", "activity"]).sum("time").reset_index()
pivot_table = df_grouped.pivot(index="hour", columns="activity", values="time").fillna(
    0
)
full_hours = pd.DataFrame({"hour": range(24)})  # Add missing hours (0 to 23)
pivot_table = full_hours.merge(pivot_table, on="hour", how="left")
pivot_table = pivot_table.fillna(0)
fig = px.bar(
    pivot_table,
    x="hour",
    y=pivot_table.columns[1:],
    labels={"value": "Time (hours)", "hour": "Hour of the Day"},
    barmode="stack",
    color_discrete_map=colordict,
    opacity=0.6,
    hover_data={"variable": True, "value": ":.2f"},
)
fig.update_layout(
    bargap=0.05,
    barcornerradius=3,
    coloraxis_showscale=False,
    xaxis=dict(
        tickmode="array",
        tickvals=[i for i in range(24)],
        ticktext=[f"{i}:00" for i in range(24)],
        tickangle=40,
    ),
    legend=dict(x=0, y=1, title="Activity"),
)
hist_cols[1].plotly_chart(fig, key="histogram_activity_times_2")


###############################################################################
#
# Scatter plot : repartition of the times at which I finish each activity
#
###############################################################################
stats_container.markdown("### Recorded Activities")
df_tmp = df[df["date"].dt.time != datetime(1970, 1, 1, 0, 0).time()].copy()
# df_tmp = df.copy()
df_tmp["hour_of_day"] = df_tmp["date"].apply(lambda x: f"{x.hour}:{x.minute:02d}")
df_tmp["minute_of_day"] = 60 * df_tmp["date"].dt.hour + df_tmp["date"].dt.minute
df_tmp["comment"] = df_tmp["comment"].fillna(" ")
all_minutes = pd.DataFrame(
    {"minute_of_day": range(24 * 60)}
)  # Add missing minute of the day
fig = px.scatter(
    df_tmp,
    x="minute_of_day",
    y="time",
    color="activity",
    labels={"time": "Time", "hour_of_day": "Hour of the Day"},
    color_discrete_map=colordict,
    opacity=0.6,
    hover_name="activity",
    hover_data=dict(
        activity=False, minute_of_day=False, time=True, hour_of_day=True, comment=True
    ),
)
fig.update_traces(marker_size=10)
fig.update_layout(
    coloraxis_showscale=False,
    xaxis_range=[0, 24 * 60],
    xaxis=dict(
        tickmode="array",
        tickvals=[i for i in range(0, 24 * 60, 30)],
        ticktext=[f"{i//60}:{i%60:02d}" for i in range(0, 24 * 60, 30)],
        tickangle=40,
    ),
    legend=dict(x=0, y=1, title="Activity"),
)
stats_container.plotly_chart(fig, key="tmp")
