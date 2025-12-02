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
    page_title="Book Store Analytics",
    layout="wide",
)


def _to_df(obj):
    """
    Normalize result objects to DataFrame when possible.

    - If obj is a DataFrame -> return as is.
    - If obj is a dict (like {'date': [...], 'revenue_usd': [...]})
      -> convert to DataFrame.
    - Otherwise -> return None.
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        try:
            return pd.DataFrame(obj)
        except Exception:
            return None
    return None


def render_dataset(name: str, folder: str):
    st.header(f"Results for {name}")

    try:
        result = process_dataset(folder)
    except Exception as e:
        st.error(f"Error while processing {name}: {e}")
        return
    df = result.get("df")
    daily_revenue_raw = result.get("daily_revenue")
    top5_days_raw = result.get("top5_days")
    unique_users = result.get("unique_users", 0)
    unique_author_sets = result.get("unique_author_sets", 0)
    popular_authors = result.get("popular_authors", [])
    best_key = result.get("best_buyer_key")
    best_revenue = result.get("best_buyer_revenue")
    best_aliases = result.get("best_buyer_aliases")

    top5_days = _to_df(top5_days_raw)
    daily_revenue = _to_df(daily_revenue_raw)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Top 5 Days by Revenue (YYYY-MM-DD)")
        if top5_days is not None and not top5_days.empty:
            top5_display = top5_days.copy()

            if "date" in top5_display.columns:
                top5_display["date"] = top5_display["date"].astype(str)
                top5_display = top5_display.rename(
                    columns={"date": "Date"}
                )
            if "revenue_usd" in top5_display.columns:
                top5_display = top5_display.rename(
                    columns={"revenue_usd": "Revenue (USD)"}
                )

            st.dataframe(top5_display, use_container_width=True)
        else:
            st.write("No revenue data available.")

    with col2:
        st.subheader("Unique Users")
        st.metric("Count", value=int(unique_users))

        st.subheader("Unique Author Sets")
        st.metric("Count", value=int(unique_author_sets))

    with col3:
        st.subheader("Most Popular Author(s)")
        if popular_authors:
            for a in popular_authors:
                st.write(f"- {a}")
        else:
            st.write("No author revenue statistics available.")

        st.subheader("Best Buyer (All user_id aliases)")
        if best_key is not None and best_revenue is not None:
            st.write(f"User key: **{int(best_key)}**")
            try:
                st.write(f"Total revenue: **{float(best_revenue):.2f} USD**")
            except Exception:
                st.write(f"Total revenue: **{best_revenue} USD**")

            st.write(f"Aliases (user_id values): {best_aliases}")
        else:
            st.write("No buyer data available.")

    st.markdown("---")

    st.subheader("Daily Revenue Chart (USD)")
    if daily_revenue is not None and not daily_revenue.empty:
        chart_df = daily_revenue.copy()

        if "date" in chart_df.columns:
            chart_df["date"] = pd.to_datetime(chart_df["date"])
            chart_df = chart_df.set_index("date")

        if "revenue_usd" in chart_df.columns:
            st.line_chart(chart_df["revenue_usd"])
        else:
            numeric_cols = chart_df.select_dtypes("number")
            if not numeric_cols.empty:
                st.line_chart(numeric_cols)
            else:
                st.write("No numeric revenue data to display.")
    else:
        st.write("No daily revenue data to display.")


def main():
    st.title("Book Store Analytics")

    tab1, tab2, tab3 = st.tabs(["DATA1", "DATA2", "DATA3"])
    with tab1:
        render_dataset("DATA1", DATASETS["DATA1"])
    with tab2:
        render_dataset("DATA2", DATASETS["DATA2"])
    with tab3:
        render_dataset("DATA3", DATASETS["DATA3"])


if __name__ == "__main__":
    main()
