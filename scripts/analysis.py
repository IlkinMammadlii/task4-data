import pandas as pd
from load_clean import load_and_clean


def _extract_orders_df(raw) -> pd.DataFrame:
    """
    Make sure we always work with a pandas DataFrame.

    `load_and_clean()` may return:
      - a single DataFrame (already merged), OR
      - a dict like {"orders": df_orders, "books": df_books} on some setups.

    This helper normalizes that so the rest of the code can safely use .empty,
    .groupby(), etc. without hitting "'dict' object has no attribute 'empty'".
    """
    # Case 1: already a DataFrame
    if isinstance(raw, pd.DataFrame):
        return raw

    # Case 2: a dict containing dataframes
    if isinstance(raw, dict):
        if "orders" in raw and isinstance(raw["orders"], pd.DataFrame):
            return raw["orders"]
        if "df" in raw and isinstance(raw["df"], pd.DataFrame):
            return raw["df"]

        # If we reach here, it's a dict but without a usable DataFrame
        raise TypeError(
            f"load_and_clean() returned a dict, but no 'orders' or 'df' "
            f"DataFrame was found. Keys: {list(raw.keys())}"
        )

    # Any other type is wrong
    raise TypeError(
        f"load_and_clean() returned unsupported type: {type(raw).__name__}"
    )


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

    # --- load raw data and normalize to a single orders DataFrame ---
    raw = load_and_clean(folder_path)
    df = _extract_orders_df(raw)

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

    # --- best buyer (by user_key, using all aliases) ---
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

        # All user_id aliases mapped to this user_key
        best_buyer_aliases = (
            df.loc[df["user_key"] == user_rev.index[0], "user_id"]
            .astype(str)
            .dropna()
            .unique()
            .tolist()
        )
        best_buyer_aliases = sorted(best_buyer_aliases)

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
