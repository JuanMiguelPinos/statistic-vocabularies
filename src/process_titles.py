import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.extract_geo import normalize_geography
from src.extract_time import extract_time_interval
from src.extract_vocabulary import (
    build_global_table_vocabulary,
    is_valid_vocabulary_term,
    normalize_term,
)


MONTH_WORDS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
    "Janvier",
    "Février",
    "Fevrier",
    "Mars",
    "Avril",
    "Mai",
    "Juin",
    "Juillet",
    "Août",
    "Aout",
    "Septembre",
    "Octobre",
    "Novembre",
    "Décembre",
    "Decembre",
]

MONTH_ALTERNATION = "|".join(
    sorted(
        (re.escape(month) for month in MONTH_WORDS),
        key=len,
        reverse=True,
    )
)

YEAR_EXPRESSION = r"(?:19\d{2}|20\d{2}|2100)"

TIME_PATTERNS = [
    re.compile(
        rf"\(\s*{YEAR_EXPRESSION}\s*=\s*100\s*\)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b{YEAR_EXPRESSION}[-/]\d{{2}}[-/]\d{{2}}\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b{YEAR_EXPRESSION}\s*[-–/]\s*{YEAR_EXPRESSION}\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b{YEAR_EXPRESSION}[\s._-]*Q[1-4]\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\bQ[1-4][\s._-]*{YEAR_EXPRESSION}\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b{YEAR_EXPRESSION}[\s._-]*M(?:0?[1-9]|1[0-2])\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:{MONTH_ALTERNATION})\s*[-–]\s*"
        rf"(?:{MONTH_ALTERNATION})\s+{YEAR_EXPRESSION}\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:{MONTH_ALTERNATION})\s+{YEAR_EXPRESSION}\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\b{YEAR_EXPRESSION}\b",
        flags=re.IGNORECASE,
    ),
]


AMBIGUOUS_SINGLE_WORD_GEOGRAPHIES = {
    "centre",
    "center",
    "central",
    "north",
    "south",
    "east",
    "west",
    "region",
    "mainland",
}


def spans_overlap(
    start: int,
    end: int,
    occupied_spans: list[tuple[int, int]],
) -> bool:
    return any(
        start < occupied_end and end > occupied_start
        for occupied_start, occupied_end in occupied_spans
    )


def normalize_time_mention(raw_text: str) -> str | None:
    text = raw_text.strip("() ")

    base_year_match = re.search(
        YEAR_EXPRESSION,
        text,
    )

    if "=" in text and base_year_match:
        return base_year_match.group(0)

    normalized = extract_time_interval(text)

    if normalized is not None:
        return normalized

    text = re.sub(
        rf"({YEAR_EXPRESSION})\s+Q([1-4])",
        r"\1-Q\2",
        text,
        flags=re.IGNORECASE,
    )

    normalized = extract_time_interval(text)

    if normalized is not None:
        return normalized

    if base_year_match:
        return base_year_match.group(0)

    return None


def extract_time_mentions(
    title: str,
) -> list[dict[str, object]]:
    mentions: list[dict[str, object]] = []
    occupied_spans: list[tuple[int, int]] = []

    for pattern in TIME_PATTERNS:
        for match in pattern.finditer(title):
            start, end = match.span()

            if spans_overlap(
                start,
                end,
                occupied_spans,
            ):
                continue

            raw_text = match.group(0)
            normalized_time = normalize_time_mention(
                raw_text
            )

            if normalized_time is None:
                continue

            mentions.append(
                {
                    "raw_text": raw_text,
                    "normalized_time": normalized_time,
                    "start": start,
                    "end": end,
                }
            )

            occupied_spans.append((start, end))

    return sorted(
        mentions,
        key=lambda mention: int(mention["start"]),
    )


def tokenize_with_spans(
    text: str,
) -> list[tuple[str, int, int]]:
    tokens: list[tuple[str, int, int]] = []

    for match in re.finditer(
        r"\w+",
        text,
        flags=re.UNICODE,
    ):
        normalized = normalize_geography(
            match.group(0)
        )

        if not normalized:
            continue

        for token in normalized.split():
            tokens.append(
                (
                    token,
                    match.start(),
                    match.end(),
                )
            )

    return tokens


def build_title_geography_index(
    geography_dictionary: pd.DataFrame,
) -> dict[str, list[dict[str, object]]]:
    index: dict[
        str,
        list[dict[str, object]],
    ] = defaultdict(list)

    seen_aliases: set[str] = set()

    for _, row in geography_dictionary.iterrows():
        normalized_alias = str(
            row["normalized_alias"]
        ).strip()

        source = str(
            row["dictionary_source"]
        )

        if "code" in source.casefold():
            continue

        if not normalized_alias:
            continue

        alias_tokens = normalized_alias.split()

        if (
            len(alias_tokens) == 1
            and alias_tokens[0]
            in AMBIGUOUS_SINGLE_WORD_GEOGRAPHIES
        ):
            continue

        if (
            len(alias_tokens) == 1
            and len(alias_tokens[0]) <= 2
        ):
            continue

        if normalized_alias in seen_aliases:
            continue

        seen_aliases.add(normalized_alias)

        index[alias_tokens[0]].append(
            {
                "alias": row["alias"],
                "normalized_alias": normalized_alias,
                "alias_tokens": alias_tokens,
                "canonical_name": row["canonical_name"],
                "geo_code": row["geo_code"],
                "dictionary_source": source,
            }
        )

    for first_token in index:
        index[first_token].sort(
            key=lambda candidate: len(
                candidate["alias_tokens"]
            ),
            reverse=True,
        )

    return dict(index)


def extract_geography_mentions(
    title: str,
    geography_index: dict[
        str,
        list[dict[str, object]],
    ],
) -> list[dict[str, object]]:
    tokens = tokenize_with_spans(title)
    mentions: list[dict[str, object]] = []

    position = 0

    while position < len(tokens):
        current_token = tokens[position][0]

        candidates = geography_index.get(
            current_token,
            [],
        )

        selected_candidate = None

        for candidate in candidates:
            alias_tokens = candidate["alias_tokens"]
            length = len(alias_tokens)

            if position + length > len(tokens):
                continue

            title_tokens = [
                tokens[index][0]
                for index in range(
                    position,
                    position + length,
                )
            ]

            if title_tokens == alias_tokens:
                selected_candidate = candidate
                break

        if selected_candidate is None:
            position += 1
            continue

        length = len(
            selected_candidate["alias_tokens"]
        )

        start = tokens[position][1]
        end = tokens[position + length - 1][2]

        mentions.append(
            {
                "raw_text": title[start:end],
                "canonical_name": selected_candidate[
                    "canonical_name"
                ],
                "geo_code": selected_candidate[
                    "geo_code"
                ],
                "dictionary_source": selected_candidate[
                    "dictionary_source"
                ],
                "start": start,
                "end": end,
            }
        )

        position += length

    return mentions


def remove_text_spans(
    text: str,
    spans: list[tuple[int, int]],
) -> str:
    if not spans:
        return text

    characters = list(text)

    for start, end in spans:
        for index in range(start, end):
            characters[index] = " "

    return "".join(characters)


def clean_residual_title(title: str) -> str:
    text = title

    text = re.sub(r"[\[\](){}]", " ", text)
    text = re.sub(r"\s*[,;:]\s*", " ", text)
    text = re.sub(r"\s+-\s+", " - ", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = text.strip(" ,;:-")

    text = re.sub(
        r"\b(?:in|at|during|from|to)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" ,;:-")

    return text


def process_titles(
    titles: pd.DataFrame,
    table_filenames: set[str],
    geography_dictionary: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    normalized_columns = {
        str(column).strip().casefold(): column
        for column in titles.columns
    }

    if "filename" not in normalized_columns:
        raise ValueError(
            "The titles file does not contain a 'filename' column."
        )

    if "title" not in normalized_columns:
        raise ValueError(
            "The titles file does not contain a 'title' column."
        )

    filename_column = normalized_columns["filename"]
    title_column = normalized_columns["title"]

    canonical_filenames = {
        filename.casefold(): filename
        for filename in table_filenames
    }

    geography_index = build_title_geography_index(
        geography_dictionary
    )

    processing_records: list[
        dict[str, object]
    ] = []

    title_vocabulary_records: list[
        dict[str, object]
    ] = []

    date_records: list[
        dict[str, object]
    ] = []

    geography_records: list[
        dict[str, object]
    ] = []

    processed_filenames: set[str] = set()

    title_rows = titles.drop_duplicates(
        subset=[filename_column],
        keep="first",
    )

    for _, row in tqdm(
        title_rows.iterrows(),
        total=len(title_rows),
        desc="Processing titles",
        unit="title",
    ):
        source_filename = str(
            row[filename_column]
        ).strip()

        canonical_filename = canonical_filenames.get(
            source_filename.casefold()
        )

        if canonical_filename is None:
            continue

        title = str(
            row[title_column]
        ).strip()

        processed_filenames.add(
            canonical_filename
        )

        time_mentions = extract_time_mentions(
            title
        )

        geography_mentions = extract_geography_mentions(
            title=title,
            geography_index=geography_index,
        )

        spans = [
            (
                int(mention["start"]),
                int(mention["end"]),
            )
            for mention in (
                time_mentions
                + geography_mentions
            )
        ]

        residual_title = clean_residual_title(
            remove_text_spans(
                text=title,
                spans=spans,
            )
        )

        processing_records.append(
            {
                "filename": canonical_filename,
                "original_title": title,
                "residual_title": residual_title,
                "date_count": len(time_mentions),
                "geography_count": len(
                    geography_mentions
                ),
                "status": "ok",
            }
        )

        for mention in time_mentions:
            date_records.append(
                {
                    "filename": canonical_filename,
                    "original_title": title,
                    "raw_time": mention["raw_text"],
                    "normalized_time": mention[
                        "normalized_time"
                    ],
                }
            )

        for mention in geography_mentions:
            geography_records.append(
                {
                    "filename": canonical_filename,
                    "original_title": title,
                    "raw_geography": mention["raw_text"],
                    "canonical_name": mention[
                        "canonical_name"
                    ],
                    "geo_code": mention[
                        "geo_code"
                    ],
                    "dictionary_source": mention[
                        "dictionary_source"
                    ],
                }
            )

        if is_valid_vocabulary_term(
            residual_title
        ):
            title_vocabulary_records.append(
                {
                    "filename": canonical_filename,
                    "term": residual_title,
                    "normalized_term": normalize_term(
                        residual_title
                    ).casefold(),
                    "source": "title",
                    "columns": "title",
                    "occurrence_count": 1,
                }
            )

    missing_filenames = sorted(
        table_filenames.difference(
            processed_filenames
        )
    )

    for filename in missing_filenames:
        processing_records.append(
            {
                "filename": filename,
                "original_title": "",
                "residual_title": "",
                "date_count": 0,
                "geography_count": 0,
                "status": "missing_title",
            }
        )

    processing = pd.DataFrame(
        processing_records,
        columns=[
            "filename",
            "original_title",
            "residual_title",
            "date_count",
            "geography_count",
            "status",
        ],
    )

    title_vocabulary = pd.DataFrame(
        title_vocabulary_records,
        columns=[
            "filename",
            "term",
            "normalized_term",
            "source",
            "columns",
            "occurrence_count",
        ],
    )

    title_dates = pd.DataFrame(
        date_records,
        columns=[
            "filename",
            "original_title",
            "raw_time",
            "normalized_time",
        ],
    )

    title_geographies = pd.DataFrame(
        geography_records,
        columns=[
            "filename",
            "original_title",
            "raw_geography",
            "canonical_name",
            "geo_code",
            "dictionary_source",
        ],
    )

    return (
        processing,
        title_vocabulary,
        title_dates,
        title_geographies,
    )


def merge_pipe_values(
    values: pd.Series,
) -> str:
    parts: set[str] = set()

    for value in values.dropna():
        for part in str(value).split("|"):
            clean_part = part.strip()

            if clean_part:
                parts.add(clean_part)

    return "|".join(
        sorted(parts)
    )


def combine_table_and_title_vocabulary(
    vocabulary_without_geo: pd.DataFrame,
    title_vocabulary: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat(
        [
            vocabulary_without_geo,
            title_vocabulary,
        ],
        ignore_index=True,
    )

    if combined.empty:
        return combined

    final_vocabulary = (
        combined
        .groupby(
            [
                "filename",
                "normalized_term",
            ],
            as_index=False,
        )
        .agg(
            term=("term", "first"),
            source=(
                "source",
                merge_pipe_values,
            ),
            columns=(
                "columns",
                merge_pipe_values,
            ),
            occurrence_count=(
                "occurrence_count",
                "sum",
            ),
        )
    )

    final_vocabulary = final_vocabulary[
        [
            "filename",
            "term",
            "normalized_term",
            "source",
            "columns",
            "occurrence_count",
        ]
    ]

    return final_vocabulary.sort_values(
        by=[
            "filename",
            "normalized_term",
        ]
    ).reset_index(drop=True)


def save_title_results(
    processing: pd.DataFrame,
    title_vocabulary: pd.DataFrame,
    title_dates: pd.DataFrame,
    title_geographies: pd.DataFrame,
    final_table_vocabulary: pd.DataFrame,
    final_global_vocabulary: pd.DataFrame,
    output_directory: Path,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    processing.to_csv(
        output_directory / "title_processing.csv",
        index=False,
        encoding="utf-8",
    )

    title_vocabulary.to_csv(
        output_directory / "title_vocabulary.csv",
        index=False,
        encoding="utf-8",
    )

    title_dates.to_csv(
        output_directory / "title_dates.csv",
        index=False,
        encoding="utf-8",
    )

    title_geographies.to_csv(
        output_directory / "title_geographies.csv",
        index=False,
        encoding="utf-8",
    )

    final_table_vocabulary.to_csv(
        output_directory / "table_vocabulary_final.csv",
        index=False,
        encoding="utf-8",
    )

    final_global_vocabulary.to_csv(
        output_directory / "global_vocabulary.csv",
        index=False,
        encoding="utf-8",
    )

    print(f"Title processing results saved to: {output_directory}")