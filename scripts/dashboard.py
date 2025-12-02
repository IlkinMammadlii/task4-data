import os
import pandas as pd
import streamlit as st
from analysis import process_dataset

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATASETS = {
    "DATA1": os.path.join(BASE_DIR, "DATA1"),
    "DATA2": os.path.join(BASE_DIR, "DATA2"),
    "DATA3": os.path.join(BASE_DIR, "DATA3"),
}

st.set_page_config(
    page_title="Task 4 – Book Store Analytics",
    layout="wide",
)


def render_dataset(name: str, folder: str):
    st.header(f"Results for {name}")

    try:
        result = process_dataset(folder)
    except Exception as e:
        st.error(f"Error while processing {name}: {e}")
        return

    df = result["df"]
    daily_revenue = result["daily_revenue"]
    top5_days = result["top5_days"]
    unique_users = result["unique_users"]
    unique_author_sets = result["unique_author_sets"]
    popular_authors = result["popular_authors"]
    best_key = result["best_buyer_key"]
    best_revenue = result["best_buyer_revenue"]
    best_aliases = result["best_buyer_aliases"]

    # --- top metrics row ---
    col1, col2, col3 = st.columns(3)

    # Top 5 days
    with col1:
        st.subheader("Top 5 Days by Revenue (YYYY-MM-DD)")
        if top5_days is not None and not top5_days.empty:
            top5_display = top5_days.copy()
            top5_display["date"] = top5_display["date"].astype(str)
            top5_display = top5_display.rename(
                columns={"date": "Date", "revenue_usd": "Revenue (USD)"}
            )
            st.dataframe(top5_display, use_container_width=True)
        else:
            st.write("No revenue data available.")

    # Unique users + authors
    with col2:
        st.subheader("Unique Users")
        st.metric("Count", value=unique_users)

        st.subheader("Unique Author Sets")
        st.metric("Count", value=unique_author_sets)

    # Popular authors + best buyer
    with col3:
        st.subheader("Most Popular Author(s)")
        if popular_authors:
            for a in popular_authors:
                st.write(f"- {a}")
        else:
            st.write("No author revenue statistics available.")

        st.subheader("Best Buyer (All user_id aliases)")
        if best_key is not None:
            st.write(f"User key: **{best_key}**")
            st.write(f"Total revenue: **{best_revenue:.2f} USD**")
            st.write(f"Aliases (user_id values): {best_aliases}")
        else:
            st.write("No buyer data available.")

    st.markdown("---")

    # --- daily revenue chart ---
    st.subheader("Daily Revenue Chart (USD)")
    if daily_revenue is not None and not daily_revenue.empty:
        chart_df = daily_revenue.copy()
        chart_df["date"] = pd.to_datetime(chart_df["date"])
        chart_df = chart_df.set_index("date")
        st.line_chart(chart_df["revenue_usd"])
    else:
        st.write("No daily revenue data to display.")


def main():
    st.title("Task 4 – Book Store Analytics")

    tab1, tab2, tab3 = st.tabs(["DATA1", "DATA2", "DATA3"])
    with tab1:
        render_dataset("DATA1", DATASETS["DATA1"])
    with tab2:
        render_dataset("DATA2", DATASETS["DATA2"])
    with tab3:
        render_dataset("DATA3", DATASETS["DATA3"])


if __name__ == "__main__":
    main()
