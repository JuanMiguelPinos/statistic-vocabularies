import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

from src.io_utils import read_csv_flexible


MIN_YEAR = 1900
MAX_YEAR = 2100

YEAR_PATTERN = re.compile(
    r"^(?P<year>19\d{2}|20\d{2}|2100)(?:\.0)?$"
)

YEAR_RANGE_PATTERN = re.compile(
    r"^(?P<start>19\d{2}|20\d{2}|2100)"
    r"\s*[-–/]\s*"
    r"(?P<end>19\d{2}|20\d{2}|2100)$"
)

QUARTER_PATTERN_1 = re.compile(
    r"^(?P<year>19\d{2}|20\d{2}|2100)"
    r"[\s._-]*Q(?P<quarter>[1-4])$",
    flags=re.IGNORECASE,
)

QUARTER_PATTERN_2 = re.compile(
    r"^Q(?P<quarter>[1-4])"
    r"[\s._-]*(?P<year>19\d{2}|20\d{2}|2100)$",
    flags=re.IGNORECASE,
)

MONTH_CODE_PATTERN = re.compile(
    r"^(?P<year>19\d{2}|20\d{2}|2100)"
    r"[\s._-]*M(?P<month>0?[1-9]|1[0-2])$",
    flags=re.IGNORECASE,
)

YEAR_MONTH_PATTERN = re.compile(
    r"^(?P<year>19\d{2}|20\d{2}|2100)"
    r"[-/](?P<month>0[1-9]|1[0-2])$"
)

FULL_DATE_PATTERN = re.compile(
    r"^(?P<year>19\d{2}|20\d{2}|2100)"
    r"[-/](?P<month>0[1-9]|1[0-2])"
    r"[-/](?P<day>0[1-9]|[12]\d|3[01])$"
)

MONTH_NAMES = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
    "janvier": "01",
    "fevrier": "02",
    "mars": "03",
    "avril": "04",
    "mai": "05",
    "juin": "06",
    "juillet": "07",
    "aout": "08",
    "septembre": "09",
    "octobre": "10",
    "novembre": "11",
    "decembre": "12",
}

NAMED_PERIODS = {
    ("january", "march"): "Q1",
    ("april", "june"): "Q2",
    ("july", "september"): "Q3",
    ("october", "december"): "Q4",
    ("janvier", "mars"): "Q1",
    ("avril", "juin"): "Q2",
    ("juillet", "septembre"): "Q3",
    ("octobre", "decembre"): "Q4",
    ("january", "june"): "H1",
    ("july", "december"): "H2",
    ("janvier", "juin"): "H1",
    ("juillet", "decembre"): "H2",
}

MONTH_NAME_PATTERN = re.compile(
    r"^(?P<month>[^\d]+?)\s+(?P<year>19\d{2}|20\d{2}|2100)$"
)

NAMED_PERIOD_PATTERN = re.compile(
    r"^(?P<start>[^\d]+?)\s*[-–]\s*"
    r"(?P<end>[^\d]+?)\s+"
    r"(?P<year>19\d{2}|20\d{2}|2100)$"
)


def clean_header_value(value: object) -> str:
    text = str(value).strip()

    if text.lower().startswith("unnamed:"):
        return ""

    return text


def extract_time_interval(value: object) -> str | None:
    text = clean_header_value(value)

    if not text:
        return None

    match = FULL_DATE_PATTERN.fullmatch(text)

    if match:
        return (
            f"{match.group('year')}-"
            f"{match.group('month')}-"
            f"{match.group('day')}"
        )

    match = YEAR_RANGE_PATTERN.fullmatch(text)

    if match:
        start = int(match.group("start"))
        end = int(match.group("end"))

        if MIN_YEAR <= start <= MAX_YEAR and MIN_YEAR <= end <= MAX_YEAR:
            return f"{start}-{end}"

    match = YEAR_PATTERN.fullmatch(text)

    if match:
        year = int(match.group("year"))

        if MIN_YEAR <= year <= MAX_YEAR:
            return str(year)

    match = QUARTER_PATTERN_1.fullmatch(text)

    if match:
        return (
            f"{match.group('year')}-Q"
            f"{match.group('quarter')}"
        )

    match = QUARTER_PATTERN_2.fullmatch(text)

    if match:
        return (
            f"{match.group('year')}-Q"
            f"{match.group('quarter')}"
        )

    match = MONTH_CODE_PATTERN.fullmatch(text)

    if match:
        month = int(match.group("month"))

        return (
            f"{match.group('year')}-M"
            f"{month:02d}"
        )

    match = YEAR_MONTH_PATTERN.fullmatch(text)

    if match:
        return (
            f"{match.group('year')}-"
            f"{match.group('month')}"
        )

    normalized_text = unidecode(text).lower().strip()

    match = NAMED_PERIOD_PATTERN.fullmatch(normalized_text)

    if match:
        start_month = match.group("start").strip()
        end_month = match.group("end").strip()

        period = NAMED_PERIODS.get(
            (start_month, end_month)
        )

        if period:
            return f"{match.group('year')}-{period}"

    match = MONTH_NAME_PATTERN.fullmatch(normalized_text)

    if match:
        month_name = match.group("month").strip()
        month_number = MONTH_NAMES.get(month_name)

        if month_number:
            return f"{match.group('year')}-{month_number}"

    return None


def extract_time_intervals_from_columns(
    columns: Iterable[object],
) -> list[str]:
    intervals: list[str] = []
    seen: set[str] = set()

    for column in columns:
        interval = extract_time_interval(column)

        if interval is not None and interval not in seen:
            intervals.append(interval)
            seen.add(interval)

    return intervals


def build_table_time_vocabulary(
    table_files: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interval_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []

    for table_path in tqdm(
        table_files,
        desc="Extracting dates",
        unit="table",
    ):
        try:
            dataframe = read_csv_flexible(
                table_path,
                nrows=0,
            )

            intervals = extract_time_intervals_from_columns(
                dataframe.columns
            )

            summary_records.append(
                {
                    "filename": table_path.name,
                    "time_interval_count": len(intervals),
                    "status": "ok",
                    "error": "",
                }
            )

            for position, interval in enumerate(intervals):
                interval_records.append(
                    {
                        "filename": table_path.name,
                        "time_interval": interval,
                        "position": position,
                    }
                )

        except Exception as exc:
            summary_records.append(
                {
                    "filename": table_path.name,
                    "time_interval_count": 0,
                    "status": "error",
                    "error": str(exc),
                }
            )

    intervals_dataframe = pd.DataFrame(
        interval_records,
        columns=[
            "filename",
            "time_interval",
            "position",
        ],
    )

    summary_dataframe = pd.DataFrame(
        summary_records,
        columns=[
            "filename",
            "time_interval_count",
            "status",
            "error",
        ],
    )

    return intervals_dataframe, summary_dataframe


def save_time_results(
    intervals: pd.DataFrame,
    summary: pd.DataFrame,
    output_directory: Path,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    intervals_path = output_directory / "table_dates.csv"
    summary_path = output_directory / "table_dates_summary.csv"

    intervals.to_csv(
        intervals_path,
        index=False,
        encoding="utf-8",
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8",
    )

    print(f"Time extraction results saved to: {output_directory}")