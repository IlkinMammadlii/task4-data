# analysis.py

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from load_clean import load_and_clean


# ---- price parsing helpers ------------------------------------------------

EUR_TO_USD = 1.1  # rough conversion, enough for analytics


def _parse_price_to_usd(value) -> float | None:
    """
    Take raw unit_price like:
        '67.0 €', 'EUR26.99', '30 $', 'USD 24.99',
        '€43¢75', '$49.50', '41.50$', '19.99 $', ...
    and return a float in USD.
    """
    if pd.isna(value):
        return None

    s = str(value).strip()
    if not s:
        return None

    lower = s.lower()

    # detect currency
    if "eur" in lower or "€" in lower:
        currency = "EUR"
    elif "usd" in lower or "$" in lower:
        currency = "USD"
    else:
        # assume USD if unknown
        currency = "USD"

    # normalize weird separators like "€43¢75" -> "43.75"
    # first replace cent sign with a dot
    s_clean = s.replace("¢", ".")
    # extract all number chunks (allow comma or dot)
    nums = re.findall(r"\d+(?:[.,]\d+)?", s_clean)
    if not nums:
        return None

    if len(nums) == 1:
        num_str = nums[0]
    else:
        # e.g. '43' and '75' -> '43.75'
        num_str = nums[0] + "." + nums[1]

    num_str = num_str.replace(",", ".")
    try:
        amount = float(num_str)
    except ValueError:
        return None

    if currency == "EUR":
        return amount * EUR_TO_USD
    else:
        return amount


# ---- author helpers -------------------------------------------------------


def _normalize_authors(author_str: str | None) -> str | None:
    """
    Turn raw author string into a canonical "authors_set":

    'Rep. Heath Stiedemann, Gino Welch, Haydee X' ->
        'Gino Welch; Haydee X; Rep. Heath Stiedemann'
    """
    if pd.isna(author_str) or author_str is None:
        return None

    s = str(author_str).strip()
    if not s:
        return None

    # split on comma, '&', 'and'
    parts = re.split(r",|&| and ", s)
    names = sorted({p.strip() for p in parts if p.strip()})

    if not names:
        return None

    return "; ".join(names)


# ---- main processing ------------------------------------------------------


def _attach_books_authors(df: pd.DataFrame, folder_path: str) -> pd.DataFrame:
    """
    If df does not have 'authors_set', read books.yaml directly from the
    DATA* folder and merge authors info.
    """
    if "authors_set" in df.columns:
        return df

    books_path = Path(folder_path) / "books.yaml"
    if not books_path.exists():
        return df

    with open(books_path, "r", encoding="utf-8") as f:
        books_list = yaml.safe_load(f)

    if not books_list:
        return df

    books_df = pd.json_normalize(books_list)

    # handle possible Ruby-style keys like ':id', ':author'
    rename_map = {}
    for col in books_df.columns:
        if col.startswith(":"):
            rename_map[col] = col[1:]
    if rename_map:
        books_df = books_df.rename(columns=rename_map)

    # we only need id + author
    if "id" not in books_df.columns:
        return df
    author_col = "author" if "author" in books_df.columns else None
    if author_col is None:
        return df

    books_df["authors_set"] = books_df[author_col].apply(_normalize_authors)

    books_df_small = books_df[["id", "authors_set"]].copy()

    # merge into orders dataframe on book_id
    df = df.merge(
        books_df_small,
        left_on="book_id",
        right_on="id",
        how="left",
        suffixes=("", "_book"),
    )

    # drop auxiliary id from books
    df = df.drop(columns=["id_book"], errors="ignore")

    return df


def process_dataset(folder_path: str) -> dict:
    """
    Load one DATA* folder, clean it and return all metrics needed by dashboard.py.

    Returned keys:
      - top5_days: DataFrame[date, paid_price]
      - unique_users: int | None
      - unique_author_sets: int | None
      - most_popular_authors: DataFrame[authors_set, paid_price] | None
      - best_buyer_aliases: dict | None
    """
    # 1) load cleaned dataframe ---------------------------------------------
    df = load_and_clean(folder_path).copy()
    result: dict[str, object] = {}

    # 2) ensure we have proper datetime + date ------------------------------
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    else:
        raise KeyError("Data must contain 'timestamp' or 'date' column")

    # 3) recompute numeric unit_price_usd + paid_price ----------------------
    # this ignores whatever load_clean tried before, to get rid of NaN mess
    if "unit_price" in df.columns:
        df["unit_price_usd"] = df["unit_price"].apply(_parse_price_to_usd)
    elif "unit_price_usd" in df.columns:
        # already there
        df["unit_price_usd"] = pd.to_numeric(df["unit_price_usd"], errors="coerce")
    else:
        df["unit_price_usd"] = 0.0

    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    else:
        df["quantity"] = 0

    df["paid_price"] = df["unit_price_usd"] * df["quantity"]
    df["paid_price"] = df["paid_price"].fillna(0.0)

    # 4) attach authors from books.yaml if missing --------------------------
    df = _attach_books_authors(df, folder_path)

    # 5) TOP 5 DAYS BY REVENUE ----------------------------------------------
    daily_rev = (
        df.groupby("date", dropna=False)["paid_price"]
        .sum(min_count=1)
        .fillna(0)
        .reset_index()
    )
    daily_rev["date"] = daily_rev["date"].astype(str)

    result["top5_days"] = daily_rev.sort_values(
        "paid_price", ascending=False
    ).head(5)

    # 6) UNIQUE USERS -------------------------------------------------------
    user_col = "user_key" if "user_key" in df.columns else "user_id"
    if user_col in df.columns:
        result["unique_users"] = int(df[user_col].nunique())
    else:
        result["unique_users"] = None

    # 7) UNIQUE AUTHOR SETS -------------------------------------------------
    if "authors_set" in df.columns:
        result["unique_author_sets"] = int(df["authors_set"].nunique())
    else:
        result["unique_author_sets"] = None

    # 8) MOST POPULAR AUTHORS (by total revenue) ---------------------------
    if "authors_set" in df.columns:
        author_rev = (
            df.groupby("authors_set", dropna=False)["paid_price"]
            .sum(min_count=1)
            .fillna(0)
            .reset_index()
        )
        result["most_popular_authors"] = author_rev.sort_values(
            "paid_price", ascending=False
        )
    else:
        result["most_popular_authors"] = None

    # 9) BEST BUYER (merge all aliases for same user_key/user_id) ----------
    if user_col in df.columns:
        user_rev = (
            df.groupby(user_col)["paid_price"]
            .sum(min_count=1)
            .fillna(0)
            .reset_index()
        )

        if not user_rev.empty:
            best_row = user_rev.sort_values(
                "paid_price", ascending=False
            ).iloc[0]

            best_key = best_row[user_col]
            best_total = float(best_row["paid_price"])

            aliases: list[str] = []
            if "user_id" in df.columns:
                aliases = (
                    df.loc[df[user_col] == best_key, "user_id"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                aliases = sorted(aliases)

            result["best_buyer_aliases"] = {
                "user_key": str(best_key),
                "total_revenue": best_total,
                "aliases": aliases,
            }
        else:
            result["best_buyer_aliases"] = None
    else:
        result["best_buyer_aliases"] = None

    return result
