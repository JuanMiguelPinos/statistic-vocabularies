import csv
import re
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.extract_time import extract_time_interval
from src.io_utils import read_csv_flexible


NUMERIC_PATTERN = re.compile(
    r"^[+-]?"
    r"(?:"
    r"\d+(?:[.,]\d+)?"
    r"|"
    r"\d{1,3}(?:[\s,]\d{3})+(?:[.,]\d+)?"
    r")$"
)

MISSING_VALUES = {
    "",
    ":",
    "-",
    "--",
    "..",
    "...",
    "na",
    "n/a",
    "nan",
    "<na>",
    "null",
    "none",
}

MALFORMED_QUOTE_FILES = {
    "prc_dap12.csv",
    "prc_dap13.csv",
    "prc_dap14.csv",
    "prc_dap15.csv",
}


def normalize_term(value: object) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.replace(
        "\u00a0",
        " ",
    )

    text = text.replace(
        "\r",
        " ",
    )

    text = text.replace(
        "\n",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if "\\Time" in text:
        text = text.split(
            "\\Time",
            maxsplit=1,
        )[0].strip()

    text = text.strip("\"' ")

    return text


def is_numeric_value(value: object) -> bool:
    text = normalize_term(value)

    if not text:
        return False

    return NUMERIC_PATTERN.fullmatch(
        text
    ) is not None


def is_valid_vocabulary_term(
    value: object,
) -> bool:
    text = normalize_term(value)

    if not text:
        return False

    if text.casefold() in MISSING_VALUES:
        return False

    if text.casefold().startswith(
        "unnamed:"
    ):
        return False

    if is_numeric_value(text):
        return False

    if extract_time_interval(text) is not None:
        return False

    return True


def find_semantic_column_count(
    columns: list[object],
) -> int:
    for index, column in enumerate(columns):
        if extract_time_interval(column) is not None:
            return index

    for index, column in enumerate(columns):
        if is_numeric_value(column):
            return index

    return len(columns)


def register_term(
    terms: dict[str, dict[str, object]],
    filename: str,
    value: object,
    source: str,
    column_name: str,
    occurrence_count: int = 1,
) -> None:
    term = normalize_term(value)

    if not is_valid_vocabulary_term(term):
        return

    key = term.casefold()

    if key not in terms:
        terms[key] = {
            "filename": filename,
            "term": term,
            "normalized_term": key,
            "sources": set(),
            "columns": set(),
            "occurrence_count": 0,
        }

    terms[key]["sources"].add(source)
    terms[key]["columns"].add(column_name)

    terms[key]["occurrence_count"] += int(
        occurrence_count
    )


def terms_to_records(
    terms: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    for information in terms.values():
        records.append(
            {
                "filename": information[
                    "filename"
                ],
                "term": information[
                    "term"
                ],
                "normalized_term": information[
                    "normalized_term"
                ],
                "source": "|".join(
                    sorted(
                        information["sources"]
                    )
                ),
                "columns": "|".join(
                    sorted(
                        information["columns"]
                    )
                ),
                "occurrence_count": information[
                    "occurrence_count"
                ],
            }
        )

    records.sort(
        key=lambda record: str(
            record["normalized_term"]
        )
    )

    return records


def register_dataframe_values(
    dataframe: pd.DataFrame,
    semantic_columns: list[object],
    filename: str,
    terms: dict[str, dict[str, object]],
) -> None:
    for column in semantic_columns:
        if column not in dataframe.columns:
            continue

        clean_column = normalize_term(column)

        value_counts = dataframe[
            column
        ].value_counts(
            dropna=False
        )

        for value, count in value_counts.items():
            register_term(
                terms=terms,
                filename=filename,
                value=value,
                source="cell",
                column_name=clean_column,
                occurrence_count=int(count),
            )


def extract_table_strings(
    dataframe: pd.DataFrame,
    filename: str,
) -> tuple[list[dict[str, object]], int]:
    columns = list(
        dataframe.columns
    )

    semantic_column_count = (
        find_semantic_column_count(
            columns
        )
    )

    semantic_columns = columns[
        :semantic_column_count
    ]

    terms: dict[
        str,
        dict[str, object],
    ] = {}

    for column in semantic_columns:
        clean_column = normalize_term(
            column
        )

        register_term(
            terms=terms,
            filename=filename,
            value=clean_column,
            source="header",
            column_name=clean_column,
        )

    register_dataframe_values(
        dataframe=dataframe,
        semantic_columns=semantic_columns,
        filename=filename,
        terms=terms,
    )

    return (
        terms_to_records(terms),
        semantic_column_count,
    )


def extract_table_strings_chunked(
    table_path: Path,
    chunk_size: int,
) -> tuple[
    list[dict[str, object]],
    int,
    int,
]:
    csv_read_options = {}

    if table_path.name.casefold() in MALFORMED_QUOTE_FILES:
        csv_read_options["quoting"] = csv.QUOTE_NONE

    header_dataframe = read_csv_flexible(
        table_path,
        nrows=0,
        **csv_read_options,
    )

    columns = list(
        header_dataframe.columns
    )

    semantic_column_count = (
        find_semantic_column_count(
            columns
        )
    )

    semantic_columns = columns[
        :semantic_column_count
    ]

    terms: dict[
        str,
        dict[str, object],
    ] = {}

    for column in semantic_columns:
        clean_column = normalize_term(
            column
        )

        register_term(
            terms=terms,
            filename=table_path.name,
            value=clean_column,
            source="header",
            column_name=clean_column,
        )

    if semantic_column_count == 0:
        return (
            terms_to_records(terms),
            semantic_column_count,
            0,
        )

    semantic_column_positions = list(
        range(semantic_column_count)
    )

    chunks = read_csv_flexible(
        table_path,
        usecols=semantic_column_positions,
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
        **csv_read_options,
    )

    chunks_processed = 0

    if isinstance(chunks, pd.DataFrame):
        chunks_iterator = [chunks]
    else:
        chunks_iterator = chunks

    for chunk in chunks_iterator:
        chunks_processed += 1

        chunk_columns = list(
            chunk.columns
        )

        register_dataframe_values(
            dataframe=chunk,
            semantic_columns=chunk_columns,
            filename=table_path.name,
            terms=terms,
        )

    return (
        terms_to_records(terms),
        semantic_column_count,
        chunks_processed,
    )


def build_table_string_vocabulary(
    table_files: list[Path],
    chunk_size: int = 20_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    vocabulary_records: list[
        dict[str, object]
    ] = []

    summary_records: list[
        dict[str, object]
    ] = []

    for table_path in tqdm(
        table_files,
        desc="Extracting S(t)",
        unit="table",
    ):
        try:
            (
                records,
                semantic_column_count,
                chunks_processed,
            ) = extract_table_strings_chunked(
                table_path=table_path,
                chunk_size=chunk_size,
            )

            vocabulary_records.extend(
                records
            )

            summary_records.append(
                {
                    "filename": (
                        table_path.name
                    ),
                    "file_size_bytes": (
                        table_path.stat().st_size
                    ),
                    "semantic_column_count": (
                        semantic_column_count
                    ),
                    "chunks_processed": (
                        chunks_processed
                    ),
                    "distinct_string_count": (
                        len(records)
                    ),
                    "status": "ok",
                    "error": "",
                }
            )

        except Exception as exc:
            summary_records.append(
                {
                    "filename": (
                        table_path.name
                    ),
                    "file_size_bytes": (
                        table_path.stat().st_size
                        if table_path.exists()
                        else 0
                    ),
                    "semantic_column_count": 0,
                    "chunks_processed": 0,
                    "distinct_string_count": 0,
                    "status": "error",
                    "error": str(exc),
                }
            )

    vocabulary = pd.DataFrame(
        vocabulary_records,
        columns=[
            "filename",
            "term",
            "normalized_term",
            "source",
            "columns",
            "occurrence_count",
        ],
    )

    summary = pd.DataFrame(
        summary_records,
        columns=[
            "filename",
            "file_size_bytes",
            "semantic_column_count",
            "chunks_processed",
            "distinct_string_count",
            "status",
            "error",
        ],
    )

    return vocabulary, summary


def build_global_table_vocabulary(
    table_vocabulary: pd.DataFrame,
) -> pd.DataFrame:
    if table_vocabulary.empty:
        return pd.DataFrame(
            columns=[
                "term",
                "normalized_term",
                "table_count",
                "total_occurrences",
            ]
        )

    global_vocabulary = (
        table_vocabulary
        .groupby(
            "normalized_term",
            as_index=False,
        )
        .agg(
            term=(
                "term",
                "first",
            ),
            table_count=(
                "filename",
                "nunique",
            ),
            total_occurrences=(
                "occurrence_count",
                "sum",
            ),
        )
    )

    global_vocabulary = (
        global_vocabulary[
            [
                "term",
                "normalized_term",
                "table_count",
                "total_occurrences",
            ]
        ]
    )

    global_vocabulary = (
        global_vocabulary
        .sort_values(
            by=[
                "table_count",
                "total_occurrences",
                "normalized_term",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return global_vocabulary


def save_vocabulary_results(
    table_vocabulary: pd.DataFrame,
    global_vocabulary: pd.DataFrame,
    summary: pd.DataFrame,
    output_directory: Path,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_path = (
        output_directory
        / "table_strings.csv"
    )

    global_path = (
        output_directory
        / "global_table_vocabulary.csv"
    )

    summary_path = (
        output_directory
        / "table_strings_summary.csv"
    )

    table_vocabulary.to_csv(
        table_path,
        index=False,
        encoding="utf-8",
    )

    global_vocabulary.to_csv(
        global_path,
        index=False,
        encoding="utf-8",
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8",
    )

    print(f"Vocabulary results saved to: {output_directory}")