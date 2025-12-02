import os
import re
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError: 
    yaml = None

EUR_TO_USD = 1.07


def clean_timestamp(ts):
    """Normalize messy timestamp strings before parsing."""
    if not isinstance(ts, str):
        return ts

    ts = ts.strip()

    ts = re.sub(
        r"\b([aA][mM]|[pP][mM])\.?\b",
        lambda m: m.group(1).upper(),
        ts,
    )
    ts = re.sub(
        r"([ap])\.?\s?m\.?",
        lambda m: m.group(1).upper() + "M",
        ts,
        flags=re.IGNORECASE,
    )

    ts = ts.title()

    ts = ts.replace(";", " ")
    ts = ts.replace(",", " ")

    ts = re.sub(r"\s+", " ", ts)

    ts = ts.rstrip(".")

    return ts


def parse_timestamp(ts):
    """Convert a single raw timestamp to pandas.Timestamp."""
    if pd.isna(ts):
        return pd.NaT

    raw = str(ts)
    cleaned = clean_timestamp(raw)

    try:
        return pd.to_datetime(
            cleaned,
            utc=False,
            errors="raise",
            dayfirst=True,  
        )
    except Exception as e:
        raise ValueError(f"Unknown timestamp: {raw} → {cleaned}") from e

def parse_price_to_usd(value):
    """
    Parse strings like:
    '67.0 €', 'EUR26.99', '30 $', 'USD 24.99', '€43¢75', '$ 32.25', ...
    into float USD.
    """
    if pd.isna(value):
        return np.nan

    s = str(value).strip()
    if not s:
        return np.nan

    original = s
    upper = s.upper()
    if "USD" in upper or "$" in s:
        currency = "USD"
    elif "EUR" in upper or "€" in s:
        currency = "EUR"
    else:
        currency = "EUR"

    s_num = original
    for token in ["USD", "EUR", "$", "€"]:
        s_num = s_num.replace(token, "")

    s_num = s_num.replace("¢", ".")
    s_num = s_num.replace(",", ".")

    s_num = re.sub(r"[^0-9.]", "", s_num)

    if s_num.count(".") > 1:
        parts = s_num.split(".")
        s_num = "".join(parts[:-1]) + "." + parts[-1]

    if s_num == "":
        return np.nan

    try:
        val = float(s_num)
    except Exception:
        return np.nan

    if currency == "USD":
        return val
    else:
        return val * EUR_TO_USD


def _load_books(books_path: str) -> pd.DataFrame:
    """Load books from CSV or YAML and normalize to [book_id, author_set]."""
    if books_path is None:
        return pd.DataFrame(columns=["book_id", "author_set"])

    lower = books_path.lower()

    if lower.endswith(".csv"):
        books = pd.read_csv(books_path)
    elif lower.endswith(".yml") or lower.endswith(".yaml"):
        if yaml is None:
            raise ImportError(
                "PyYAML is required to read books.yml. Install it with 'pip install pyyaml'."
            )
        with open(books_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
        books = pd.DataFrame(data)
    else:
        return pd.DataFrame(columns=["book_id", "author_set"])
    
    orig_cols = list(books.columns)
    norm_cols = []
    for c in orig_cols:
        c_str = str(c).strip()
        c_str = c_str.lstrip(":")  
        norm_cols.append(c_str)
    books.columns = norm_cols

    if "book_id" not in books.columns and "id" in books.columns:
        books = books.rename(columns={"id": "book_id"})

    author_col = None
    for c in books.columns:
        base = c.strip().lower()
        if base in ("author", "authors"):
            author_col = c
            break

    if author_col is not None:
        books["author_set"] = (
            books[author_col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )
    else:
        books["author_set"] = np.nan

    if "book_id" not in books.columns:
        books["book_id"] = np.nan

    books["book_id"] = pd.to_numeric(books["book_id"], errors="coerce").astype("Int64")

    return books[["book_id", "author_set"]]


def load_and_clean(folder_path: str) -> pd.DataFrame:
    """
    Load users (optional), books, and orders from DATA1/2/3,
    clean them, and return a unified DataFrame with:
      user_id, user_key, book_id, timestamp, date,
      unit_price_usd, paid_price_usd, author_set, revenue_usd
    """
    users_path = None
    books_path = None
    orders_path = None

    for fname in os.listdir(folder_path):
        lower = fname.lower()
        full = os.path.join(folder_path, fname)

        if lower.startswith("user") and lower.endswith(".csv"):
            users_path = full
        elif lower.startswith("book") and (
            lower.endswith(".csv") or lower.endswith(".yml") or lower.endswith(".yaml")
        ):
            books_path = full
        elif lower.startswith("order") and (
            lower.endswith(".csv") or lower.endswith(".parquet")
        ):
            orders_path = full

    if orders_path is None:
        raise FileNotFoundError(f"No orders file found in {folder_path}")

    if users_path and os.path.exists(users_path):
        _ = pd.read_csv(users_path)  

    books = _load_books(books_path)

    if orders_path.lower().endswith(".csv"):
        orders = pd.read_csv(orders_path)
    else:
        orders = pd.read_parquet(orders_path)

    if "timestamp" not in orders.columns:
        raise KeyError("orders file has no 'timestamp' column")

    orders["timestamp"] = orders["timestamp"].apply(parse_timestamp)
    orders["date"] = orders["timestamp"].dt.date

    if "unit_price" not in orders.columns:
        raise KeyError("orders file has no 'unit_price' column")

    orders["unit_price_usd"] = orders["unit_price"].apply(parse_price_to_usd)
    orders["quantity"] = pd.to_numeric(
        orders.get("quantity", 1), errors="coerce"
    ).fillna(0).astype(float)
    orders["paid_price_usd"] = orders["unit_price_usd"] * orders["quantity"]

    if "user_id" not in orders.columns:
        raise KeyError("orders file has no 'user_id' column")
    orders["user_key"] = orders["user_id"].astype(str)

    if "book_id" in orders.columns:
        orders["book_id"] = pd.to_numeric(
            orders["book_id"], errors="coerce"
        ).astype("Int64")
    else:
        orders["book_id"] = pd.arrays.IntegerArray([pd.NA] * len(orders))

    df = orders.merge(books, on="book_id", how="left")

    df["revenue_usd"] = df["paid_price_usd"]

    return df
