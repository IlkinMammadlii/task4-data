import pandas as pd
from load_clean import load_and_clean


def process_dataset(folder_path: str) -> dict:
    """
    Load + clean dataset and compute metrics for the dashboard.

    Returns dict with keys:
      - df
      - daily_revenue  (DataFrame[date, revenue_usd])
      - top5_days      (DataFrame[date, revenue_usd])
      - unique_users   (int)
      - unique_author_sets (int)
      - popular_authors (list[str])
      - best_buyer_key (str or None)
      - best_buyer_revenue (float)
      - best_buyer_aliases (list[str])
    """
    df = load_and_clean(folder_path)
    if df.empty:
        raise ValueError("Dataset is empty")

    # --- daily revenue ---
    daily_revenue = (
        df.groupby("date", as_index=False)["revenue_usd"]
        .sum()
        .sort_values("date")
    )

    # --- top 5 days by revenue ---
    top5_days = (
        daily_revenue.sort_values("revenue_usd", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

    # --- unique users ---
    unique_users = int(df["user_key"].nunique())

    # --- authors ---
    if "author_set" in df.columns:
        unique_author_sets = int(df["author_set"].dropna().nunique())
        author_rev = (
            df.dropna(subset=["author_set"])
            .groupby("author_set")["revenue_usd"]
            .sum()
            .sort_values(ascending=False)
        )
        popular_authors = author_rev.head(3).index.tolist()
    else:
        unique_author_sets = 0
        popular_authors = []

    # --- best buyer ---
    user_rev = (
        df.groupby("user_key")["revenue_usd"]
        .sum()
        .sort_values(ascending=False)
    )

    if user_rev.empty:
        best_buyer_key = None
        best_buyer_revenue = 0.0
        best_buyer_aliases = []
    else:
        best_buyer_key = str(user_rev.index[0])
        best_buyer_revenue = float(user_rev.iloc[0])
        best_buyer_aliases = sorted(
            df.loc[df["user_key"] == best_buyer_key, "user_id"]
            .astype(str)
            .unique()
            .tolist()
        )

    return {
        "df": df,
        "daily_revenue": daily_revenue,
        "top5_days": top5_days,
        "unique_users": unique_users,
        "unique_author_sets": unique_author_sets,
        "popular_authors": popular_authors,
        "best_buyer_key": best_buyer_key,
        "best_buyer_revenue": best_buyer_revenue,
        "best_buyer_aliases": best_buyer_aliases,
    }
