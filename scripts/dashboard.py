import streamlit as st
import pandas as pd
from pathlib import Path

from load_clean import load_and_clean


st.set_page_config(
    page_title="Task 4 – Book Store Analytics",
    layout="wide",
)


DATA_ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "DATA1": DATA_ROOT / "DATA1",
    "DATA2": DATA_ROOT / "DATA2",
    "DATA3": DATA_ROOT / "DATA3",
}


def render_dataset(dataset_name: str, dataset_path: Path) -> None:
    st.header(f"Results for {dataset_name}")

    try:
        result = load_and_clean(str(dataset_path))
    except Exception as e:
        st.error(f"Error while processing {dataset_name}: {e}")
        return

    orders = result["orders"]

    # ----------------- METRIC CARDS -----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Unique Users", int(result["unique_users"]))

    with col2:
        if result["has_author_info"] and result["unique_author_sets"] is not None:
            st.metric("Unique Author Sets", int(result["unique_author_sets"]))
        else:
            st.metric("Unique Author Sets", "N/A")

    with col3:
        st.metric("Total Orders", int(len(orders)))

    st.markdown("---")

    # ----------------- DAILY REVENUE CHART -----------------
    st.subheader("Daily Revenue")

    daily_rev = (
        orders.groupby("date", as_index=False)["paid_price"]
        .sum(min_count=1)
        .dropna(subset=["paid_price"])
        .sort_values("date")
    )

    if not daily_rev.empty:
        # make sure date column is proper datetime for plotting
        daily_rev["date"] = pd.to_datetime(daily_rev["date"])
        daily_rev_chart = daily_rev.set_index("date")["paid_price"]
        st.line_chart(daily_rev_chart, use_container_width=True)
    else:
        st.info("No revenue data available for daily chart.")

    st.markdown("---")

    # ----------------- TOP 5 DAYS BY REVENUE -----------------
    st.subheader("Top 5 Days by Revenue (YYYY-MM-DD)")

    top5 = result["top5_days"].copy()
    if not top5.empty:
        top5["date"] = pd.to_datetime(top5["date"]).dt.strftime("%Y-%m-%d")
        top5 = top5.rename(columns={"paid_price": "revenue_usd"})
        st.dataframe(top5, use_container_width=True, hide_index=True)
    else:
        st.info("No revenue days to display.")

    st.markdown("---")

    # ----------------- MOST POPULAR AUTHORS -----------------
    st.subheader("Most Popular Author(s)")

    if not result["has_author_info"]:
        st.warning("Author information is not available for this dataset.")
    else:
        mpa = result["most_popular_authors"]
        if mpa is None or mpa.empty:
            st.info("No author revenue statistics available.")
        else:
            df_auth = mpa.rename(
                columns={"author_set": "authors", "paid_price": "total_revenue_usd"}
            )
            st.dataframe(df_auth, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ----------------- BEST BUYER (WITH ALIASES) -----------------
    st.subheader("Best Buyer (All user_id aliases)")

    best_buyer = result["best_buyer"]
    if best_buyer is None:
        st.info("No buyer information available.")
    else:
        aliases = best_buyer.get("aliases") or []
        aliases_display = "[" + ", ".join(str(a) for a in aliases) + "]"

        st.write(
            f"**User key:** `{best_buyer['user_key']}`  "
            f"— **Total revenue (USD):** `{best_buyer['total_revenue']:.2f}`"
        )
        st.write("**Aliases (user_id values) as array:**")
        st.code(aliases_display, language="python")


def main():
    st.title("Task 4 – Book Store Analytics")

    tabs = st.tabs(list(DATASETS.keys()))
    for tab, (name, path) in zip(tabs, DATASETS.items()):
        with tab:
            render_dataset(name, path)


if __name__ == "__main__":
    main()
