import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


# --- Currency assumptions ---
EUR_TO_USD = 1.08


# ---------- TIMESTAMP CLEANING ----------

def clean_timestamp(ts: Any) -> Any:
    """Normalize the timestamp string before parsing."""
    if pd.isna(ts):
        return ts
    s = str(ts).strip()

    # unify whitespace
    s = re.sub(r"\s+", " ", s)

    # normalize AM/PM variants
    s = re.sub(
        r"\b([aA][mM]|[pP][mM])\.?\b",
        lambda m: m.group(1).upper(),
        s,
    )
    s = re.sub(r"([aA])\.?\s?m\.?", "AM", s)
    s = re.sub(r"([pP])\.?\s?m\.?", "PM", s)

    # normalize month names / abbreviations
    s = s.title()

    # replace weird separators
    s = s.replace(";", " ")
    s = s.replace(" ,", ",")
    s = re.sub(r"\s*,\s*", ",", s)

    # remove trailing dot
    s = s.rstrip(".")

    return s


def parse_timestamp(ts: Any) -> Any:
    """Parse many crazy formats into pandas Timestamp. Return NaT on failure."""
    if pd.isna(ts):
        return pd.NaT

    s = clean_timestamp(ts)

    # Try normal parse
    try:
        return pd.to_datetime(s, utc=False, errors="raise")
    except Exception:
        pass

    # Try European day-first
    try:
        return pd.to_datetime(s, utc=False, errors="raise", dayfirst=True)
    except Exception:
        pass

    # Give up
    return pd.NaT


# ---------- PRICE CLEANING ----------

def price_to_usd(raw: Any) -> float:
    """Convert messy unit_price string into a float in USD."""
    if pd.isna(raw):
        return np.nan

    s = str(raw).strip()
    if not s:
        return np.nan

    lower = s.lower()
    is_usd = "usd" in lower or "$" in s
    is_eur = "eur" in lower or "€" in s

    cleaned = (
        s.replace("USD", "")
        .replace("usd", "")
        .replace("EUR", "")
        .replace("eur", "")
        .replace("$", "")
        .replace("€", "")
    )

    # handle "€43¢75"
    cleaned = cleaned.replace("¢", ".")
    cleaned = cleaned.replace(",", ".")

    m = re.search(r"(\d+(\.\d+)?)", cleaned)
    if not m:
        return np.nan

    amount = float(m.group(1))

    if not is_usd and not is_eur:
        is_eur = True  # default to EUR

    if is_usd:
        amount_usd = amount
    else:
        amount_usd = amount * EUR_TO_USD

    return float(amount_usd)


# ---------- BOOKS / AUTHORS HELPERS ----------

def load_books(path: Path) -> pd.DataFrame:
    books_file = None
    for pattern in ("books.yaml", "books.yml", "books.csv", "books.parquet"):
        candidate = path / pattern
        if candidate.exists():
            books_file = candidate
            break

    if books_file is None:
        return pd.DataFrame()

    if books_file.suffix in {".yaml", ".yml"}:
        import yaml

        with open(books_file, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        books = pd.json_normalize(raw)
    elif books_file.suffix == ".csv":
        books = pd.read_csv(books_file)
    else:
        books = pd.read_parquet(books_file)

    # strip leading ":" from YAML keys
    books.columns = [c.lstrip(":") for c in books.columns]

    if "authors" not in books.columns and "author" in books.columns:
        books["authors"] = books["author"]

    if "authors" in books.columns:
        books["author_set"] = (
            books["authors"]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan})
        )

    return books


# ---------- MAIN LOAD + CLEAN FUNCTION ----------

def load_and_clean(root: str) -> Dict[str, Any]:
    root_path = Path(root)

    # --- load orders ---
    orders_file = None
    for pattern in ("orders.parquet", "orders.csv"):
        candidate = root_path / pattern
        if candidate.exists():
            orders_file = candidate
            break
    if orders_file is None:
        raise FileNotFoundError(f"No orders file found in {root}")

    if orders_file.suffix == ".parquet":
        orders = pd.read_parquet(orders_file)
    else:
        orders = pd.read_csv(orders_file)

    # --- load users ---
    users_file = None
    for pattern in ("users.csv", "users.parquet"):
        candidate = root_path / pattern
        if candidate.exists():
            users_file = candidate
            break
    if users_file is None:
        raise FileNotFoundError(f"No users file found in {root}")

    if users_file.suffix == ".parquet":
        users = pd.read_parquet(users_file)
    else:
        users = pd.read_csv(users_file)

    users.columns = [c.strip().lower() for c in users.columns]

    # --- load books ---
    books = load_books(root_path)

    # ---------- CLEAN ORDERS ----------
    orders.columns = [c.strip().lower() for c in orders.columns]

    required_cols = {"id", "user_id", "book_id", "quantity", "unit_price", "timestamp"}
    missing = required_cols - set(orders.columns)
    if missing:
        raise KeyError(f"Missing columns in orders: {missing}")

    # 1) parse timestamp values
    orders["timestamp"] = orders["timestamp"].apply(parse_timestamp)

    # *** HARD CAST TO DATETIME to fix .dt error ***
    orders["timestamp"] = pd.to_datetime(orders["timestamp"], errors="coerce")

    # drop rows with invalid timestamp
    orders = orders[orders["timestamp"].notna()].copy()

    # date column
    orders["date"] = orders["timestamp"].dt.date

    # quantity
    orders["quantity"] = (
        pd.to_numeric(orders["quantity"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # price -> USD
    orders["unit_price_usd"] = orders["unit_price"].apply(price_to_usd)

    # paid_price
    orders["paid_price"] = orders["quantity"] * orders["unit_price_usd"]

    # ---------- USER KEYS (ALIASES) ----------
    users_for_merge = users.rename(columns={"id": "user_id"})
    if "email" in users_for_merge.columns:
        merged = orders.merge(
            users_for_merge[["user_id", "email"]],
            on="user_id",
            how="left",
            validate="m:1",
        )
        merged["user_key"] = merged["email"].where(
            merged["email"].notna(), merged["user_id"].astype(str)
        )
    else:
        merged = orders.copy()
        merged["user_key"] = merged["user_id"].astype(str)

    orders = merged

    # ---------- METRICS ----------

    # Top 5 days by revenue (drop full-NaN days)
    daily_rev = (
        orders.groupby("date", as_index=False)["paid_price"]
        .sum(min_count=1)
        .dropna(subset=["paid_price"])
        .sort_values("paid_price", ascending=False)
        .head(5)
    )

    unique_users = orders["user_key"].nunique()

    # Author metrics
    has_author_info = False
    unique_author_sets = None
    most_popular_authors = None

    if not books.empty and "author_set" in books.columns:
        has_author_info = True
        unique_author_sets = books["author_set"].dropna().nunique()

        if "id" in books.columns:
            books_for_merge = books[["id", "author_set"]].rename(
                columns={"id": "book_id"}
            )
            ob = orders.merge(books_for_merge, on="book_id", how="left")

            author_rev = (
                ob.dropna(subset=["author_set", "paid_price"])
                .groupby("author_set", as_index=False)["paid_price"]
                .sum()
                .sort_values("paid_price", ascending=False)
            )

            if not author_rev.empty:
                most_popular_authors = author_rev.head(3)
        else:
            has_author_info = False

    # Best buyer
    buyer_rev = (
        orders.dropna(subset=["paid_price"])
        .groupby("user_key", as_index=False)["paid_price"]
        .sum()
        .sort_values("paid_price", ascending=False)
    )

    if buyer_rev.empty:
        best_buyer = None
    else:
        top_key = buyer_rev.iloc[0]["user_key"]
        top_total = float(buyer_rev.iloc[0]["paid_price"])
        aliases_raw = (
            orders.loc[orders["user_key"] == top_key, "user_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        best_buyer = {
            "user_key": top_key,
            "total_revenue": top_total,
            "aliases": aliases_raw,
        }

    return {
        "orders": orders,
        "top5_days": daily_rev,
        "unique_users": unique_users,
        "has_author_info": has_author_info,
        "unique_author_sets": unique_author_sets,
        "most_popular_authors": most_popular_authors,
        "best_buyer": best_buyer,
    }
