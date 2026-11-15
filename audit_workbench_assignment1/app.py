from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import json
import re
import sqlite3
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Audit Workbench",
    page_icon="📊",
    layout="wide",
)

PREVIEW_ROWS = 100
BASE_DIR = Path(__file__).parent
HISTORY_DB = BASE_DIR / "audit_history.db"

# Layout only — no hard-coded colours, so text stays readable in light AND dark theme.
# Colours come from .streamlit/config.toml (clean light theme).
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }
        div.stButton > button, div.stDownloadButton > button { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PERSISTENT AUDIT HISTORY (SQLite)
# ============================================================
def get_history_connection() -> sqlite3.Connection:
    """Open the local history database."""
    conn = sqlite3.connect(HISTORY_DB)
    conn.row_factory = sqlite3.Row
    return conn


def create_history_table() -> None:
    """Create the history table once. Existing history is kept."""
    conn = get_history_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            staff_name TEXT NOT NULL,
            event_type TEXT NOT NULL,
            method TEXT,
            filename TEXT,
            sheet_name TEXT,
            header_row INTEGER,
            population INTEGER,
            selected_rows INTEGER,
            details_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def add_history(
    *,
    staff_name: str,
    event_type: str,
    method: Optional[str] = None,
    filename: Optional[str] = None,
    sheet_name: Optional[str] = None,
    header_row: Optional[int] = None,
    population: Optional[int] = None,
    selected_rows: Optional[int] = None,
    details: Optional[dict] = None,
) -> None:
    """Save one audit action permanently."""
    conn = get_history_connection()
    cursor = conn.execute(
        """
        INSERT INTO audit_history (
            created_at, staff_name, event_type, method, filename,
            sheet_name, header_row, population, selected_rows, details_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            staff_name.strip() or "Unknown Staff",
            event_type,
            method,
            filename,
            sheet_name,
            header_row,
            population,
            selected_rows,
            json.dumps(details or {}, ensure_ascii=False, default=str),
        ),
    )
    conn.commit()
    # Remember which entries were made in THIS session (this browser tab)
    st.session_state.setdefault("session_history_ids", []).append(cursor.lastrowid)
    conn.close()


def get_history() -> list[sqlite3.Row]:
    """Return newest history first."""
    conn = get_history_connection()
    rows = conn.execute(
        "SELECT * FROM audit_history ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows


def clear_history() -> None:
    conn = get_history_connection()
    conn.execute("DELETE FROM audit_history")
    conn.commit()
    conn.close()


create_history_table()


# ============================================================
# FILE LOADING
# ============================================================
@st.cache_data(show_spinner=False)
def get_excel_sheet_names(file_bytes: bytes) -> list[str]:
    excel_file = pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl")
    return excel_file.sheet_names


@st.cache_data(show_spinner=False)
def load_file(
    file_bytes: bytes,
    filename: str,
    header_row: int = 1,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load CSV/XLSX into pandas without sorting, deduplicating or changing rows.
    """
    suffix = Path(filename).suffix.lower()
    pandas_header = header_row - 1

    if suffix == ".csv":
        return pd.read_csv(
            BytesIO(file_bytes),
            header=pandas_header,
            low_memory=False,
        )

    if suffix == ".xlsx":
        return pd.read_excel(
            BytesIO(file_bytes),
            sheet_name=sheet_name if sheet_name is not None else 0,
            header=pandas_header,
            engine="openpyxl",
        )

    raise ValueError("Unsupported file type. Please upload a CSV or XLSX file.")


# ============================================================
# SAMPLING METHODS
# ============================================================
def sample_odd_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[::2].copy()


def sample_top_n(df: pd.DataFrame, n: int) -> Tuple[pd.DataFrame, Optional[str]]:
    if n <= 0:
        raise ValueError("N must be greater than 0.")

    if n > len(df):
        return (
            df.copy(),
            f"You requested {n:,} rows, but only {len(df):,} rows are available. "
            "All available rows were returned.",
        )

    return df.head(n).copy(), None


def sample_random(
    df: pd.DataFrame,
    n: int,
    seed: int,
) -> Tuple[pd.DataFrame, Optional[str]]:
    if n <= 0:
        raise ValueError("The requested sample size must be greater than 0.")

    warning = None
    effective_n = n

    if n > len(df):
        effective_n = len(df)
        warning = (
            f"You requested {n:,} rows, but only {len(df):,} rows are available. "
            "All available rows were returned."
        )

    result = df.sample(
        n=effective_n,
        replace=False,
        random_state=seed,
    ).sort_index()

    return result.copy(), warning


def sample_systematic(
    df: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, dict, Optional[str]]:
    """
    Population = number of source rows.
    Interval = Population / Sample size.
    Random start is chosen inside the first interval.
    Then one row is selected every interval.
    """
    population = len(df)

    if sample_size <= 0:
        raise ValueError("Sample size must be greater than 0.")
    if population == 0:
        raise ValueError("The uploaded dataset contains no data rows.")

    warning = None

    if sample_size > population:
        warning = (
            f"You requested {sample_size:,} rows, but the population contains "
            f"only {population:,}. All available rows were returned."
        )
        metadata = {
            "population": population,
            "sample_size": population,
            "interval": 1.0,
            "random_start": 1.0,
        }
        return df.copy(), metadata, warning

    interval = population / sample_size
    rng = np.random.default_rng(seed)
    start_zero_based = rng.uniform(0, interval)

    positions = np.floor(
        start_zero_based + np.arange(sample_size) * interval
    ).astype(int)
    positions = np.clip(positions, 0, population - 1)

    result = df.iloc[positions].copy()

    metadata = {
        "population": population,
        "sample_size": len(result),
        "interval": interval,
        "random_start": start_zero_based + 1,
        "selected_positions": (positions + 1).tolist(),
    }

    return result, metadata, warning


# ============================================================
# AUDIT TESTS
# ============================================================
def numeric_series_from_columns(
    df: pd.DataFrame,
    mode: str,
    column_1: str,
    column_2: Optional[str] = None,
) -> pd.Series:
    first = pd.to_numeric(df[column_1], errors="coerce")

    if mode == "One amount column":
        return first

    if column_2 is None:
        raise ValueError("Please select the second amount column.")

    second = pd.to_numeric(df[column_2], errors="coerce")
    return first.fillna(0) + second.fillna(0)


def benford_filter(
    df: pd.DataFrame,
    amounts: pd.Series,
    selected_digits: list[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Calculate the Benford table, then filter by auditor-selected digits."""
    numeric = pd.to_numeric(amounts, errors="coerce")
    valid_mask = numeric.notna() & (numeric >= 1)
    valid = numeric.loc[valid_mask]

    if valid.empty:
        raise ValueError("No numeric amounts of 1 or greater were available for Benford testing.")

    magnitude = np.floor(np.log10(valid))
    first_digits_valid = np.floor(valid / np.power(10.0, magnitude)).astype(int)
    first_digits_valid = first_digits_valid.clip(1, 9)

    first_digits = pd.Series(pd.NA, index=df.index, dtype="Int64")
    first_digits.loc[valid.index] = first_digits_valid.astype("int64")

    digits = np.arange(1, 10)
    counts = first_digits_valid.value_counts().reindex(digits, fill_value=0).sort_index()
    actual = counts / len(first_digits_valid)
    expected = np.log10(1 + (1 / digits))

    summary = pd.DataFrame(
        {
            "Digit": digits,
            "Count": counts.to_numpy(),
            "Expected %": expected * 100,
            "Actual %": actual.to_numpy() * 100,
            "Actual - Expected (pp)": (actual.to_numpy() - expected) * 100,
        }
    )

    result = df.loc[first_digits.isin(selected_digits)].copy()
    return result, summary, len(valid)


def weekend_filter(df: pd.DataFrame, date_column: str) -> Tuple[pd.DataFrame, int]:
    parsed = pd.to_datetime(df[date_column], errors="coerce")
    valid_dates = int(parsed.notna().sum())
    mask = parsed.dt.dayofweek.isin([5, 6])
    return df.loc[mask].copy(), valid_dates


def public_holiday_filter(
    df: pd.DataFrame,
    date_column: str,
    holiday_text: str,
) -> Tuple[pd.DataFrame, list[str], int]:
    raw_lines = [line.strip() for line in holiday_text.splitlines() if line.strip()]

    if not raw_lines:
        raise ValueError("Please enter at least one public holiday date.")

    parsed_holidays = pd.to_datetime(raw_lines, errors="coerce")
    invalid_lines = [
        raw_lines[i]
        for i, value in enumerate(parsed_holidays)
        if pd.isna(value)
    ]

    holidays = pd.DatetimeIndex(parsed_holidays.dropna()).normalize().unique()
    if len(holidays) == 0:
        raise ValueError("None of the holiday dates could be read.")

    parsed_dates = pd.to_datetime(df[date_column], errors="coerce")
    valid_dates = int(parsed_dates.notna().sum())
    mask = parsed_dates.dt.normalize().isin(holidays)

    return df.loc[mask].copy(), invalid_lines, valid_dates


def round_million_filter(
    df: pd.DataFrame,
    amounts: pd.Series,
) -> Tuple[pd.DataFrame, int]:
    numeric = pd.to_numeric(amounts, errors="coerce")
    valid_numeric = int(numeric.notna().sum())
    remainder = np.mod(numeric.abs(), 1_000_000)

    mask = (
        numeric.notna()
        & (numeric != 0)
        & np.isclose(remainder, 0, atol=1e-6)
    )

    return df.loc[mask].copy(), valid_numeric


# ============================================================
# DOWNLOAD / DISPLAY HELPERS
# ============================================================
def sanitize_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._")
    return stem or "sample"


@st.cache_data(show_spinner=False)
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


@st.cache_data(show_spinner=False)
def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Selected Data")
    return buffer.getvalue()


def preview_with_original_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Show ALL rows (scrollable table) with 1-based original row positions."""
    preview = df.copy()
    preview.index = preview.index + 1
    preview.index.name = "Original row position"
    return preview


def result_slug(method: str, selected_count: int, seed: Optional[int] = None) -> str:
    mapping = {
        "Odd-numbered rows": "odd_rows",
        "Weekend transactions": "weekends",
        "Public holiday transactions": "public_holidays",
        "000'000 ending transactions": "round_million",
        "Benford test": "benford_flagged",
    }
    if method == "Top N rows":
        return f"top_{selected_count}"
    if method == "Limit rows":
        return f"first_{selected_count}"
    if method == "Random rows":
        return f"random_{selected_count}_seed{seed}"
    if method == "Systematic random":
        return f"systematic_{selected_count}_seed{seed}"
    return mapping.get(method, "selected")


# ============================================================
# KEEP FILE + SETTINGS WHEN SWITCHING WORKSPACE <-> HISTORY
# ============================================================
# Streamlit forgets a widget's value when its page is not shown.
# Re-saving these keys at the top of every run stops that.
PERSISTENT_WIDGET_KEYS = [
    "worksheet", "header_row", "je_tests", "sampling_method",
    "amount_col", "amount_col_2", "date_col",
    "benford_digits", "holiday_text",
    "row_count", "materiality",
]


def keep_widget_state() -> None:
    for key in PERSISTENT_WIDGET_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def init_state(key: str, default) -> None:
    """Set a widget's starting value once (instead of value=/default=)."""
    if key not in st.session_state:
        st.session_state[key] = default


def valid_choice(key: str, options: list) -> None:
    """Forget a remembered choice that no longer exists (e.g. a new file)."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


class SavedUpload:
    """Stored copy of the uploaded file — works like Streamlit's UploadedFile."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def on_upload_change() -> None:
    """Runs only when the user uploads a new file or clicks the X to remove it."""
    file = st.session_state.get("source_file_uploader")
    if file is None:
        st.session_state.pop("saved_upload", None)
    else:
        st.session_state["saved_upload"] = SavedUpload(file.name, file.getvalue())


def guess_amount_column(key: str, df: pd.DataFrame) -> None:
    """Pre-select a debit/credit/amount column (first time only)."""
    if key in st.session_state and st.session_state[key] in df.columns:
        return
    pattern = re.compile(r"debit|credit|amount|дебет|кредит|дүн|мөнгө", re.IGNORECASE)
    for col in df.columns:
        if pattern.search(str(col)):
            st.session_state[key] = col
            return
    for col in df.columns:
        if pd.to_numeric(df[col], errors="coerce").notna().mean() > 0.5:
            st.session_state[key] = col
            return


def guess_date_column(key: str, df: pd.DataFrame) -> None:
    """Pre-select a date column (first time only)."""
    if key in st.session_state and st.session_state[key] in df.columns:
        return
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            st.session_state[key] = col
            return
    pattern = re.compile(r"date|огноо|өдөр", re.IGNORECASE)
    for col in df.columns:
        if pattern.search(str(col)):
            st.session_state[key] = col
            return


# ============================================================
# HISTORY (one box: the latest saved run)
# ============================================================
HISTORY_LOG_TITLES = {
    "Systematic random": "Systematic Sampling",
    "Random rows": "Random Record Sampling",
    "Top N rows": "Top Records Extraction",
    "Odd-numbered rows": "Direct Extraction",
    "Benford test": "Benford's Law",
    "Public holiday transactions": "Direct Extraction",
    "Weekend transactions": "Direct Extraction",
    "000'000 ending transactions": "Direct Extraction",
}


def format_log_datetime(value: Optional[str]) -> str:
    """2026-08-13T15:30:50 -> 8/13/2026 3:30:50 PM (same style as IDEA)."""
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return str(value)
    hour_12 = dt.hour % 12 or 12
    am_pm = "AM" if dt.hour < 12 else "PM"
    return f"{dt.month}/{dt.day}/{dt.year} {hour_12}:{dt.minute:02d}:{dt.second:02d} {am_pm}"


def format_log_value(value) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) if value else "-"
    return str(value)


def build_run_summary(row: sqlite3.Row) -> str:
    """A short summary of one saved run (no step details)."""
    details = json.loads(row["details_json"] or "{}")
    sheet = row["sheet_name"]
    source = row["filename"] or ""
    if sheet and sheet != "CSV":
        source = f"{source}  [{sheet}]"

    def line(label: str, value) -> str:
        return f"{label + ':':<18}{format_log_value(value)}"

    lines = [
        f"{format_log_datetime(row['created_at'])}     {row['staff_name']}",
        "-" * 54,
        line("File", source),
        line("Methods", row["method"]),
        line("Records", row["population"]),
        line("Records selected", row["selected_rows"]),
    ]

    mat = details.get("materiality")
    if mat and mat.get("materiality"):
        ratio = mat.get("non_material_vs_material_pct")
        lines += [
            line("Materiality", f"{mat['materiality']:,.2f}"),
            line("Material", f"{mat['material_rows']:,} rows · {mat['material_amount']:,.2f}"),
            line("Non-material", f"{mat['non_material_rows']:,} rows · {mat['non_material_amount']:,.2f}"
                 + (f"  ({ratio:.2f}% of material)" if ratio is not None else "")),
        ]

    if details.get("tiers"):
        lines.append(line(
            "Selected by",
            "  ·  ".join(f"{t['Tests matched']} tests: {t['Selected']}" for t in details["tiers"]),
        ))

    coverage = details.get("coverage")
    if coverage:
        pct = coverage.get("coverage_pct")
        lines += [
            line("Amount field", coverage.get("amount_field")),
            line(
                "Selected amount",
                f"{coverage.get('selected_amount', 0):,.2f}  of  {coverage.get('total_amount', 0):,.2f}",
            ),
            line("Coverage", f"{pct:.2f}%" if pct is not None else "n/a"),
        ]
    return "\n".join(lines)


def show_history_page() -> None:
    st.title("History")

    runs = [row for row in get_history() if row["event_type"] != "Import"]  # newest first
    if not runs:
        st.info("No run saved yet. Run a Journal entry test or Sampling and click “Save to History”.")
        return

    st.caption("Latest saved run")
    st.code(build_run_summary(runs[0]), language=None, wrap_lines=True)

    if st.toggle("Show older runs", value=False, key="history_show_all"):
        older = runs[1:]
        if older:
            st.code("\n\n".join(build_run_summary(r) for r in older), language=None, wrap_lines=True)
        else:
            st.caption("No older runs.")

        export_df = pd.DataFrame(
            [
                {
                    "Date/Time": r["created_at"].replace("T", " "),
                    "Staff": r["staff_name"],
                    "File": r["filename"],
                    "Sheet": r["sheet_name"],
                    "Methods": r["method"],
                    "Records": r["population"],
                    "Records selected": r["selected_rows"],
                    "Details": r["details_json"],
                }
                for r in runs
            ]
        )
        c1, c2 = st.columns(2)
        c1.download_button(
            "Download history CSV",
            data=export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="audit_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if c2.button("Delete all history", use_container_width=True):
            clear_history()
            st.rerun()


# ============================================================
# METHODS RUN TOGETHER (filters first, then sampling)
# ============================================================
MAX_METHODS = 5
DEFAULT_SAMPLE = 22  # starting value for sample size / number of rows (can be changed)
RANDOM_SEED = 42  # fixed, so the same data always gives the same sample (re-performable)

# Fixed order: filters narrow the rows first, sampling picks from what is left.
METHOD_ORDER = [
    "Benford test",
    "Public holiday transactions",
    "Weekend transactions",
    "000'000 ending transactions",
    "Odd-numbered rows",
    "Top N rows",
    "Random rows",
    "Systematic random",
]

NONE_OPTION = "— none —"

DEFAULT_HOLIDAYS = """2025-01-01
2025-01-29
2025-01-30
2025-01-31
2025-03-08
2025-06-01
2025-07-11
2025-07-12
2025-07-13
2025-07-14
2025-07-15
2025-11-21
2025-11-26
2025-12-29"""


def size_input(label: str, key: str) -> int:
    """Number of rows — starts at 22, can be changed to any number (1 or more)."""
    init_state(key, DEFAULT_SAMPLE)
    return int(
        st.number_input(
            label,
            min_value=1,
            step=1,
            key=key,
        )
    )


def amount_setting(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    """ONE amount setting, shared by Benford, 000'000 and the coverage %."""
    columns = list(df.columns)
    guess_amount_column("amount_col", df)
    valid_choice("amount_col", columns)
    c1, c2 = st.columns(2)
    column_1 = c1.selectbox("Amount column", options=columns, key="amount_col")
    second_options = [NONE_OPTION] + [c for c in columns if c != column_1]
    valid_choice("amount_col_2", second_options)
    column_2 = c2.selectbox(
        "Add a second amount column (optional)",
        options=second_options,
        key="amount_col_2",
        help="e.g. Debit + Credit",
    )
    if column_2 == NONE_OPTION:
        return "One amount column", column_1, None
    return "Add two columns", column_1, column_2


def apply_method(
    method: str,
    current: pd.DataFrame,
    settings: dict,
) -> Tuple[pd.DataFrame, dict, dict, Optional[str]]:
    """Apply one method to the rows left by the previous step.
    Returns: (rows kept, details for history, extra (Benford table), warning)"""
    details: dict = {}
    extra: dict = {}
    warning: Optional[str] = None
    mode, column_1, column_2 = settings["amount"]
    amount_details = {"amount_mode": mode, "amount_column_1": column_1, "amount_column_2": column_2}

    if current.empty:
        return current.copy(), details, extra, None

    if method == "Odd-numbered rows":
        return sample_odd_rows(current), {"rule": "positions 1, 3, 5, 7, ..."}, extra, None

    if method == "Top N rows":
        n = settings["row_count"]
        result, warning = sample_top_n(current, n)
        return result, {"requested_n": n}, extra, warning

    if method == "Random rows":
        n = settings["row_count"]
        result, warning = sample_random(current, n, RANDOM_SEED)
        details = {
            "requested_n": n,
            "seed": RANDOM_SEED,
            "replacement": False,
            "selected_original_rows": (result.index + 1).tolist(),
        }
        return result, details, extra, warning

    if method == "Systematic random":
        n = settings["row_count"]
        result, info, warning = sample_systematic(current, n, RANDOM_SEED)
        details = {
            "sample_size_requested": n,
            "seed": RANDOM_SEED,
            "interval": info["interval"],
            "random_start": info["random_start"],
            "selected_original_rows": (result.index + 1).tolist(),
        }
        return result, details, extra, warning

    if method == "Benford test":
        amounts = numeric_series_from_columns(current, mode, column_1, column_2)
        result, summary, valid_count = benford_filter(current, amounts, settings["digits"])
        extra["benford_summary"] = summary
        kept = pd.to_numeric(amounts.loc[result.index], errors="coerce")
        extra["column"] = (
            "Benford first digit",
            np.floor(kept / np.power(10.0, np.floor(np.log10(kept)))).astype(int),
        )
        details = {**amount_details, "selected_digits": settings["digits"], "valid_amounts": valid_count}
        return result, details, extra, None

    if method == "Public holiday transactions":
        result, invalid, valid_count = public_holiday_filter(current, settings["date_col"], settings["holiday_text"])
        details = {
            "date_column": settings["date_col"],
            "holiday_dates": [x.strip() for x in settings["holiday_text"].splitlines() if x.strip()],
            "valid_dates": valid_count,
        }
        if invalid:
            warning = "these holiday dates could not be read: " + ", ".join(invalid)
        extra["column"] = ("Public holiday", pd.Series("Yes", index=result.index))
        return result, details, extra, warning

    if method == "Weekend transactions":
        result, valid_count = weekend_filter(current, settings["date_col"])
        extra["column"] = ("Weekend", pd.to_datetime(result[settings["date_col"]], errors="coerce").dt.day_name())
        return result, {"date_column": settings["date_col"], "valid_dates": valid_count}, extra, None

    if method == "000'000 ending transactions":
        amounts = numeric_series_from_columns(current, mode, column_1, column_2)
        result, valid_count = round_million_filter(current, amounts)
        extra["column"] = ("000'000 ending", pd.Series("Yes", index=result.index))
        details = {**amount_details, "valid_numeric_amounts": valid_count, "rule": "amount divisible by 1,000,000"}
        return result, details, extra, None

    raise ValueError(f"Unknown method: {method}")


def amount_coverage(
    df: pd.DataFrame,
    result: pd.DataFrame,
    mode: str,
    column_1: str,
    column_2: Optional[str],
) -> dict:
    """Selected rows' amount as % of the amount of ALL rows (absolute values)."""
    amounts = numeric_series_from_columns(df, mode, column_1, column_2).abs().fillna(0)
    total = float(amounts.sum())
    selected = float(amounts.loc[result.index].sum()) if len(result) else 0.0
    field = column_1 if mode == "One amount column" else f"{column_1} + {column_2}"
    return {
        "amount_field": field,
        "total_amount": total,
        "selected_amount": selected,
        "coverage_pct": (selected / total * 100) if total else None,
        "total_rows": len(df),
        "selected_rows": len(result),
    }


def result_with_positions(result: pd.DataFrame) -> pd.DataFrame:
    """Final rows with their original row position as the first column."""
    out = result.copy()
    out.insert(0, "Original row position", result.index + 1)
    return out


def run_summary_table(
    filename: str,
    sheet_name: Optional[str],
    df: pd.DataFrame,
    steps: list[dict],
    coverage: dict,
) -> pd.DataFrame:
    rows = [
        ("File", filename),
        ("Sheet", sheet_name or "CSV"),
        ("Rows in original file", len(df)),
        ("Rows selected", coverage["selected_rows"]),
        ("Rows selected %", f"{coverage['selected_rows'] / len(df) * 100:.2f}%" if len(df) else "n/a"),
        ("Amount field", coverage["amount_field"]),
        ("Total amount (all rows)", round(coverage["total_amount"], 2)),
        ("Selected amount", round(coverage["selected_amount"], 2)),
        (
            "Amount coverage %",
            f"{coverage['coverage_pct']:.2f}%" if coverage["coverage_pct"] is not None else "n/a",
        ),
        ("", ""),
    ]
    for i, step in enumerate(steps, start=1):
        rows.append((f"Step {i}", step["method"]))
        rows.append((f"Step {i} rows in → kept", f"{step['rows_in']:,} → {step['rows_out']:,}"))
        for key, value in step["details"].items():
            if key == "selected_original_rows":
                continue
            rows.append((f"Step {i} {key}", format_log_value(value)))
    return pd.DataFrame(rows, columns=["Item", "Value"])


BENFORD_HIGHLIGHT = "background-color: #C6EFCE; color: #006100; font-weight: bold"


def benford_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Benford table in the auditor's layout: Numbers · Count · Log10 · % · Var (+ total)."""
    table = pd.DataFrame(
        {
            "Numbers": summary["Digit"].astype(str),
            "Count": summary["Count"].astype(float),
            "Log10": summary["Expected %"] / 100,
            "%": summary["Actual %"] / 100,
            "Var": (summary["Expected %"] - summary["Actual %"]) / 100,
        }
    )
    total = pd.DataFrame([{"Numbers": "Total", "Count": float(summary["Count"].sum())}])
    return pd.concat([table, total], ignore_index=True)


def style_benford(table: pd.DataFrame):
    """Var = Log10 − %. Negative Var (actual above expected) → green Var cell, as in the workpaper."""
    def highlight(row: pd.Series) -> list[str]:
        negative = pd.notna(row["Var"]) and row["Var"] < 0
        return [
            BENFORD_HIGHLIGHT if (col == "Var" and negative)
            else ("font-weight: bold" if row["Numbers"] == "Total" else "")
            for col in row.index
        ]

    return (
        table.style
        .apply(highlight, axis=1)
        .format({"Count": "{:,.0f}", "Log10": "{:.2%}", "%": "{:.2%}", "Var": "{:.2%}"}, na_rep="")
        # number formats for the Excel download
        .map(lambda _: "number-format: 0.00%", subset=["Log10", "%", "Var"])
        .map(lambda _: "number-format: #,##0", subset=["Count"])
    )


def build_excel_report(
    result: pd.DataFrame,
    summary: pd.DataFrame,
    benford: Optional[pd.DataFrame],
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        result.to_excel(writer, index=False, sheet_name="Selected rows")
        summary.to_excel(writer, index=False, sheet_name="Summary")
        if benford is not None:
            style_benford(benford_table(benford)).to_excel(writer, index=False, sheet_name="Benford summary")
    return buffer.getvalue()


def build_output_name(filename: str, steps: list[dict]) -> str:
    slugs = [
        result_slug(step["method"], step["rows_out"], step["details"].get("seed"))
        for step in steps
    ]
    return f"{sanitize_filename(filename)}_" + "_".join(slugs)


# ============================================================
# JOURNAL ENTRY TEST (overlap of the 4 tests)
# ============================================================
JE_TESTS = [
    "Benford test",
    "Weekend transactions",
    "Public holiday transactions",
    "000'000 ending transactions",
]
JE_COLUMNS = {
    "Benford test": "Benford digit",
    "Weekend transactions": "Weekend",
    "Public holiday transactions": "Public holiday",
    "000'000 ending transactions": "000'000 ending",
}
SAMPLING_METHODS = ["Systematic random", "Random rows", "Top N rows", "Odd-numbered rows"]


def journal_entry_flags(
    df: pd.DataFrame,
    tests: list[str],
    settings: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], list[str]]:
    """Run every chosen test on ALL rows (like Excel filters side by side).
    Returns: (True/False per test, value to show per test, Benford table, notes)"""
    flags = pd.DataFrame(index=df.index)
    values = pd.DataFrame(index=df.index)
    benford_summary = None
    notes: list[str] = []
    mode, column_1, column_2 = settings["amount"]

    if "Benford test" in tests:
        amounts = pd.to_numeric(numeric_series_from_columns(df, mode, column_1, column_2), errors="coerce")
        flagged, benford_summary, _ = benford_filter(df, amounts, settings["digits"])
        flags["Benford test"] = df.index.isin(flagged.index)
        kept = amounts.loc[flagged.index]
        digits = np.floor(kept / np.power(10.0, np.floor(np.log10(kept)))).astype(int)
        values["Benford test"] = pd.Series(pd.NA, index=df.index, dtype="Int64")  # blank when not flagged
        values.loc[flagged.index, "Benford test"] = digits.astype("int64")

    if "Weekend transactions" in tests:
        dates = pd.to_datetime(df[settings["date_col"]], errors="coerce")
        is_weekend = dates.dt.dayofweek.isin([5, 6])
        flags["Weekend transactions"] = is_weekend.to_numpy()
        values["Weekend transactions"] = np.where(is_weekend, dates.dt.day_name(), "")

    if "Public holiday transactions" in tests:
        flagged, invalid, _ = public_holiday_filter(df, settings["date_col"], settings["holiday_text"])
        flags["Public holiday transactions"] = df.index.isin(flagged.index)
        values["Public holiday transactions"] = np.where(flags["Public holiday transactions"], "Yes", "")
        if invalid:
            notes.append("These holiday dates could not be read: " + ", ".join(invalid))

    if "000'000 ending transactions" in tests:
        amounts = numeric_series_from_columns(df, mode, column_1, column_2)
        flagged, _ = round_million_filter(df, amounts)
        flags["000'000 ending transactions"] = df.index.isin(flagged.index)
        values["000'000 ending transactions"] = np.where(flags["000'000 ending transactions"], "Yes", "")

    return flags, values, benford_summary, notes


def select_by_overlap(
    df: pd.DataFrame,
    flags: pd.DataFrame,
    amounts_abs: pd.Series,
    row_count: int,
) -> Tuple[pd.Index, pd.Series, pd.DataFrame]:
    """Take rows matching the MOST tests first (4 → 3 → 2 → 1) until row_count is reached.
    Inside a group that has more rows than needed, the largest amounts are taken first."""
    score = flags.sum(axis=1).astype(int)
    order = pd.DataFrame(
        {"score": score, "amount": amounts_abs, "position": np.arange(len(df))},
        index=df.index,
    )
    candidates = order[order["score"] > 0].sort_values(
        ["score", "amount", "position"], ascending=[False, False, True]
    )
    selected = candidates.index[:row_count]

    total_tests = flags.shape[1]
    tiers = pd.DataFrame(
        [
            {
                "Tests matched": f"{k} of {total_tests}",
                "Rows in file": int((score == k).sum()),
                "Selected": int((score.loc[selected] == k).sum()),
            }
            for k in range(total_tests, 0, -1)
        ]
    )
    return selected, score, tiers


def journal_entry_export(
    result: pd.DataFrame,
    score: pd.Series,
    values: pd.DataFrame,
    tests: list[str],
) -> pd.DataFrame:
    """Selected rows with: Original row position, Tests matched, one column per test, then the data."""
    out = pd.DataFrame(index=result.index)
    out["Original row position"] = result.index + 1
    out["Tests matched"] = score.loc[result.index].to_numpy()
    for test in tests:
        out[JE_COLUMNS[test]] = values.loc[result.index, test].to_numpy()
    for col in result.columns:
        out[col] = result[col].to_numpy()
    return out.reset_index(drop=True)


def journal_entry_summary(
    filename: str,
    sheet_name: Optional[str],
    df: pd.DataFrame,
    tests: list[str],
    settings: dict,
    tiers: pd.DataFrame,
    coverage: dict,
) -> pd.DataFrame:
    rows = [
        ("Section", "Journal entry test"),
        ("File", filename),
        ("Sheet", sheet_name or "CSV"),
        ("Rows in original file", len(df)),
        ("Tests", ", ".join(tests)),
        ("Rule", "Most tests matched first (4 → 3 → 2 → 1); largest amounts first inside a group"),
        ("Rows selected", coverage["selected_rows"]),
        ("Amount field", coverage["amount_field"]),
        ("Total amount (all rows)", round(coverage["total_amount"], 2)),
        ("Selected amount", round(coverage["selected_amount"], 2)),
        (
            "Amount coverage %",
            f"{coverage['coverage_pct']:.2f}%" if coverage["coverage_pct"] is not None else "n/a",
        ),
        ("", ""),
    ]
    for _, tier in tiers.iterrows():
        rows.append((f"Matched {tier['Tests matched']} tests", f"{tier['Rows in file']:,} in file · {tier['Selected']:,} selected"))
    if "Benford test" in tests:
        rows.append(("Benford digits", ", ".join(str(d) for d in settings["digits"])))
    if "date_col" in settings:
        rows.append(("Date column", settings["date_col"]))
    return pd.DataFrame(rows, columns=["Item", "Value"])


def show_benford_table(summary: pd.DataFrame) -> None:
    """Benford table like the Excel working paper: Numbers · Count · Log10 · % · Var."""
    table = benford_table(summary)
    negative_var = table["Var"].lt(0).fillna(False).to_numpy()
    # text version for the screen, so the Total row shows blanks (not "None")
    shown = pd.DataFrame(
        {
            "Numbers": table["Numbers"],
            "Count": [f"{v:,.0f}" for v in table["Count"]],
            "Log10": ["" if pd.isna(v) else f"{v:.2%}" for v in table["Log10"]],
            "%": ["" if pd.isna(v) else f"{v:.2%}" for v in table["%"]],
            "Var": ["" if pd.isna(v) else f"{v:.2%}" for v in table["Var"]],
        }
    )

    def highlight(row: pd.Series) -> list[str]:
        return [
            BENFORD_HIGHLIGHT if (col == "Var" and negative_var[row.name])
            else ("font-weight: bold" if row["Numbers"] == "Total" else "")
            for col in row.index
        ]

    styled = shown.style.apply(highlight, axis=1).set_properties(
        subset=["Count", "Log10", "%", "Var"], **{"text-align": "right"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("Var = Log10 − %. Green = digit appears more often than Benford expects.")


def coverage_metrics(df: pd.DataFrame, result: pd.DataFrame, coverage: dict) -> None:
    r1, r2, r3 = st.columns(3)
    r1.metric("Rows in file", f"{len(df):,}")
    r2.metric("Rows selected", f"{len(result):,}")
    r3.metric(
        "Amount coverage",
        f"{coverage['coverage_pct']:.2f}%" if coverage["coverage_pct"] is not None else "n/a",
    )
    st.caption(
        f"Selected amount {coverage['selected_amount']:,.2f} of {coverage['total_amount']:,.2f} "
        f"({coverage['amount_field']})"
    )


def save_and_download_buttons(
    excel_bytes: bytes,
    csv_bytes: bytes,
    output_name: str,
) -> bool:
    b1, b2, b3 = st.columns(3)
    save_clicked = b1.button("Save to History", type="primary", use_container_width=True)
    b2.download_button(
        "Download Excel",
        data=excel_bytes,
        file_name=f"{output_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    b3.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"{output_name}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    return save_clicked


# ============================================================
# UPLOAD (shared by both sections)
# ============================================================
def upload_section(staff_name: str) -> Optional[tuple]:
    """Returns (df, uploaded_file, sheet_name, header_row) or None."""
    with st.container(border=True):
        st.subheader("1 · Upload data")

        new_upload = st.file_uploader(
            "Upload a CSV or Excel file",
            type=["csv", "xlsx"],
            key="source_file_uploader",
            on_change=on_upload_change,
        )
        if new_upload is not None and "saved_upload" not in st.session_state:
            st.session_state["saved_upload"] = SavedUpload(new_upload.name, new_upload.getvalue())

        # Use the remembered file, so it stays when switching pages
        uploaded_file = st.session_state.get("saved_upload")

        if uploaded_file is None:
            st.info("Upload a CSV or XLSX file to begin.")
            return None

        if new_upload is None:
            keep_cols = st.columns([4, 1])
            keep_cols[0].success(f"Using uploaded file: **{uploaded_file.name}**")
            if keep_cols[1].button("Remove file", use_container_width=True):
                st.session_state.pop("saved_upload", None)
                st.rerun()

        file_bytes = uploaded_file.getvalue()
        suffix = Path(uploaded_file.name).suffix.lower()
        sheet_name = None

        option_cols = st.columns([2, 1])
        with option_cols[0]:
            if suffix == ".xlsx":
                try:
                    sheets = get_excel_sheet_names(file_bytes)
                except Exception as exc:
                    st.error(f"Could not open this Excel workbook: {exc}")
                    return None

                valid_choice("worksheet", sheets)
                sheet_name = st.selectbox("Worksheet", options=sheets, key="worksheet")
            else:
                st.caption("CSV file")

        with option_cols[1]:
            init_state("header_row", 1)
            header_row = int(
                st.number_input(
                    "Header row",
                    step=1,
                    min_value=1,
                    help="Row containing the column names.",
                    key="header_row",
                )
            )

        try:
            with st.spinner("Reading file..."):
                df = load_file(
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                    header_row=header_row,
                    sheet_name=sheet_name,
                )
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
            return None

        if len(df.columns) == 0:
            st.error("The uploaded file does not contain readable columns.")
            return None

        m1, m2 = st.columns(2)
        m1.metric("Rows", f"{len(df):,}")
        m2.metric("Columns", f"{len(df.columns):,}")
        st.write(
            f"**File:** {uploaded_file.name}"
            + (f"  ·  **Sheet:** {sheet_name}" if sheet_name else "")
        )

        # Log import only once per source configuration in the current session.
        import_key = f"{uploaded_file.name}|{sheet_name}|{header_row}|{len(df)}|{len(df.columns)}"
        if st.session_state.get("last_logged_import") != import_key:
            add_history(
                staff_name=staff_name,
                event_type="Import",
                method="Import from Excel" if suffix == ".xlsx" else "Import from CSV",
                filename=uploaded_file.name,
                sheet_name=sheet_name or "CSV",
                header_row=header_row,
                population=len(df),
                selected_rows=len(df),
                details={
                    "columns": len(df.columns),
                    "source_type": "Excel" if suffix == ".xlsx" else "CSV",
                },
            )
            st.session_state["last_logged_import"] = import_key

        st.write("**Source data** (all rows)")
        st.dataframe(preview_with_original_positions(df), use_container_width=True, height=320)

    if df.empty:
        st.warning("The file contains headers but no data rows.")
        return None

    return df, uploaded_file, sheet_name, header_row


# ============================================================
# PAGE: JOURNAL ENTRY TEST
# ============================================================
def show_journal_entry_page(staff_name: str) -> None:
    st.title("Journal entry test")
    st.caption("Benford, weekend, public holiday and 000'000 tests together — rows matching the most tests come first")

    loaded = upload_section(staff_name)
    if loaded is None:
        return
    df, uploaded_file, sheet_name, header_row = loaded

    with st.container(border=True):
        st.subheader("2 · Tests")
        init_state("je_tests", list(JE_TESTS))
        chosen = st.multiselect("Tests to combine", options=JE_TESTS, key="je_tests")
        tests = [t for t in JE_TESTS if t in chosen]

    if not tests:
        st.info("Choose at least one test.")
        return

    settings: dict = {}
    with st.container(border=True):
        st.subheader("3 · Settings")
        settings["amount"] = amount_setting(df)

        if "Public holiday transactions" in tests or "Weekend transactions" in tests:
            guess_date_column("date_col", df)
            valid_choice("date_col", list(df.columns))
            settings["date_col"] = st.selectbox("Date column", options=list(df.columns), key="date_col")

        if "Benford test" in tests:
            init_state("benford_digits", [1, 2, 5, 8, 9])
            settings["digits"] = st.multiselect(
                "Benford — first digits to include",
                options=list(range(1, 10)),
                key="benford_digits",
            )

        if "Public holiday transactions" in tests:
            init_state("holiday_text", DEFAULT_HOLIDAYS)
            with st.expander("Public holiday dates"):
                settings["holiday_text"] = st.text_area(
                    "One date per line (YYYY-MM-DD)",
                    height=220,
                    key="holiday_text",
                )

        settings["row_count"] = size_input("Number of rows", "row_count")

    try:
        flags, values, benford_summary, notes = journal_entry_flags(df, tests, settings)
        amounts_abs = numeric_series_from_columns(df, *settings["amount"]).abs().fillna(0)
        selected, score, tiers = select_by_overlap(df, flags, amounts_abs, settings["row_count"])
        result = df.loc[selected]
        coverage = amount_coverage(df, result, *settings["amount"])
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Could not run the tests: {exc}")
        return

    export_df = journal_entry_export(result, score, values, tests)
    output_name = f"{sanitize_filename(uploaded_file.name)}_journal_entry_{len(result)}"

    # ---------------- Benford table (before Results) ----------------
    results_number = 4
    if benford_summary is not None:
        with st.container(border=True):
            st.subheader("4 · Benford table")
            show_benford_table(benford_summary)
        results_number = 5

    with st.container(border=True):
        st.subheader(f"{results_number} · Results")
        coverage_metrics(df, result, coverage)
        for note in notes:
            st.caption(note)

        st.write("**Overlap** — rows matching the most tests are taken first")
        st.dataframe(tiers, use_container_width=True, hide_index=True)

        if len(result) == 0:
            st.info("No rows matched any of the selected tests.")
        else:
            st.dataframe(export_df, use_container_width=True, hide_index=True)

        summary_df = journal_entry_summary(
            uploaded_file.name, sheet_name, df, tests, settings, tiers, coverage
        )
        if save_and_download_buttons(
            build_excel_report(export_df, summary_df, benford_summary),
            export_df.to_csv(index=False).encode("utf-8-sig"),
            output_name,
        ):
            add_history(
                staff_name=staff_name,
                event_type="Audit test / sampling",
                method="Journal entry test: " + " + ".join(tests),
                filename=uploaded_file.name,
                sheet_name=sheet_name or "CSV",
                header_row=header_row,
                population=len(df),
                selected_rows=len(result),
                details={
                    "section": "Journal entry test",
                    "tests": tests,
                    "tiers": tiers.to_dict(orient="records"),
                    "selected_original_rows": (result.index + 1).tolist(),
                    "coverage": coverage,
                    "output_name": output_name,
                },
            )
            st.success("Saved to History.")


# ============================================================
# PAGE: SAMPLING
# ============================================================
def show_sampling_page(staff_name: str) -> None:
    st.title("Sampling")
    st.caption("Material items are all taken; the sample is drawn from the non-material rows only")

    loaded = upload_section(staff_name)
    if loaded is None:
        return
    df, uploaded_file, sheet_name, header_row = loaded

    with st.container(border=True):
        st.subheader("2 · Method")
        init_state("sampling_method", "Systematic random")
        method = st.radio(
            "Sampling method",
            options=SAMPLING_METHODS,
            horizontal=True,
            key="sampling_method",
            label_visibility="collapsed",
        )

    settings: dict = {}
    with st.container(border=True):
        st.subheader("3 · Settings")
        settings["amount"] = amount_setting(df)
        c1, c2 = st.columns(2)
        with c1:
            init_state("materiality", 0.0)
            settings["materiality"] = float(
                st.number_input(
                    "Materiality (amount)",
                    min_value=0.0,
                    step=1_000_000.0,
                    format="%.2f",
                    key="materiality",
                    help="Rows with amount ≥ materiality are material — all of them are selected. 0 = no material items.",
                )
            )
        with c2:
            settings["row_count"] = size_input("Number of rows (non-material sample)", "row_count")

    notes: list[str] = []
    try:
        # ---- Split material / non-material ----
        amounts_abs = numeric_series_from_columns(df, *settings["amount"]).abs().fillna(0)
        materiality = settings["materiality"]
        material_mask = (amounts_abs >= materiality) if materiality > 0 else pd.Series(False, index=df.index)
        material_df = df.loc[material_mask]
        non_material_df = df.loc[~material_mask]
        material_amount = float(amounts_abs[material_mask].sum())
        non_material_amount = float(amounts_abs[~material_mask].sum())
        materiality_info = {
            "materiality": materiality,
            "material_rows": int(material_mask.sum()),
            "material_amount": material_amount,
            "non_material_rows": int((~material_mask).sum()),
            "non_material_amount": non_material_amount,
            "non_material_vs_material_pct": (non_material_amount / material_amount * 100) if material_amount else None,
        }

        # ---- Sample from the non-material rows only ----
        sample, details, extra, warning = apply_method(method, non_material_df, settings)
        if warning:
            notes.append(warning)
        steps = [{
            "method": method,
            "rows_in": len(non_material_df),
            "rows_out": len(sample),
            "details": details,
            "extra": extra,
        }]
        row_count = settings["row_count"]
        if len(sample) > row_count:
            notes.append(f"{len(sample):,} rows matched — showing the first {row_count:,}.")
            limited = sample.head(row_count)
            steps.append({
                "method": "Limit rows",
                "rows_in": len(sample),
                "rows_out": len(limited),
                "details": {"rule": f"first {row_count:,} rows (file order)"},
                "extra": {},
            })
            sample = limited

        result = pd.concat([material_df, sample])
        coverage = amount_coverage(df, result, *settings["amount"])
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Could not run the sampling: {exc}")
        return

    output_name = build_output_name(uploaded_file.name, steps)
    if len(material_df):
        output_name = output_name.replace(
            sanitize_filename(uploaded_file.name) + "_",
            f"{sanitize_filename(uploaded_file.name)}_material_{len(material_df)}_",
            1,
        )

    # Material rows first, then the sample
    export_df = result_with_positions(result)
    export_df.insert(1, "Type", ["Material"] * len(material_df) + ["Sample"] * len(sample))

    with st.container(border=True):
        st.subheader("4 · Results")

        if materiality > 0:
            ratio = materiality_info["non_material_vs_material_pct"]
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Material ({materiality_info['material_rows']:,} rows)", f"{material_amount:,.2f}")
            m2.metric(f"Non-material ({materiality_info['non_material_rows']:,} rows)", f"{non_material_amount:,.2f}")
            m3.metric("Non-material ÷ Material", f"{ratio:.2f}%" if ratio is not None else "n/a")

        coverage_metrics(df, result, coverage)
        st.caption(f"{len(material_df):,} material + {len(sample):,} sample rows")
        for note in notes:
            st.caption(note)

        if len(result) == 0:
            st.info("No rows selected.")
        else:
            st.dataframe(export_df, use_container_width=True, hide_index=True)

        summary_df = run_summary_table(uploaded_file.name, sheet_name, df, steps, coverage)
        if materiality > 0:
            ratio = materiality_info["non_material_vs_material_pct"]
            summary_df = pd.concat(
                [
                    pd.DataFrame(
                        [
                            ("Materiality", round(materiality, 2)),
                            (f"Material amount ({materiality_info['material_rows']:,} rows)", round(material_amount, 2)),
                            (f"Non-material amount ({materiality_info['non_material_rows']:,} rows)", round(non_material_amount, 2)),
                            ("Non-material ÷ Material %", f"{ratio:.2f}%" if ratio is not None else "n/a"),
                            ("", ""),
                        ],
                        columns=["Item", "Value"],
                    ),
                    summary_df,
                ],
                ignore_index=True,
            )

        if save_and_download_buttons(
            build_excel_report(export_df, summary_df, None),
            export_df.to_csv(index=False).encode("utf-8-sig"),
            output_name,
        ):
            add_history(
                staff_name=staff_name,
                event_type="Audit test / sampling",
                method="Sampling: " + method,
                filename=uploaded_file.name,
                sheet_name=sheet_name or "CSV",
                header_row=header_row,
                population=len(df),
                selected_rows=len(result),
                details={
                    "section": "Sampling",
                    "materiality": materiality_info,
                    "steps": [
                        {k: step[k] for k in ("method", "rows_in", "rows_out", "details")}
                        for step in steps
                    ],
                    "coverage": coverage,
                    "output_name": output_name,
                },
            )
            st.success("Saved to History.")


# ============================================================
# APP ENTRY POINT
# ============================================================
def main() -> None:
    keep_widget_state()

    with st.sidebar:
        st.title("Audit Workbench")
        page = st.radio("Navigate", ["Journal entry test", "Sampling", "History"])
        st.divider()
        staff_name = st.text_input(
            "Staff name",
            value=st.session_state.get("staff_name", "Staff MN"),
            help="This name is stored in the audit history.",
        )
        st.session_state["staff_name"] = staff_name
        st.caption("History is stored locally in audit_history.db")

    if page == "History":
        show_history_page()
    elif page == "Sampling":
        show_sampling_page(staff_name)
    else:
        show_journal_entry_page(staff_name)


if __name__ == "__main__":
    main()
