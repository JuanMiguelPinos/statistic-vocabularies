import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from unidecode import unidecode


UNIT_COLUMN_NAMES = {
    "unit",
    "unit of measure",
    "measurement unit",
    "currency",
    "price unit",
    "energy unit",
    "volume unit",
    "weight unit",
}

MEASURE_COLUMN_NAMES = {
    "indicator",
    "indicators",
    "statistical information",
    "statistical indicator",
    "measure",
    "measures",
    "observed variable",
    "economic indicator",
    "economical indicator for structural business statistics",
    "health indicator",
    "environmental indicator",
    "indicator used in euro med",
}

NON_MEASURE_COLUMN_EXPRESSIONS = {
    "categorisation",
    "categorization",
    "classification",
    "category",
    "type of",
    "breakdown",
}

UNIT_PATTERNS = [
    re.compile(
        r"\b(?:percent|percentage|per cent)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:euro|euros|dollar|dollars|currency)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:kilogram|kilograms|gram|grams|tonne|tonnes)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:litre|litres|liter|liters)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:metre|metres|meter|meters|kilometre|kilometres)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:hectare|hectares)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:watt|watts|kilowatt|kilowatts|megawatt|megawatts)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:joule|joules|kwh|mwh|gwh)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:purchasing power standard|pps)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bindex points?\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:persons?|inhabitants?|enterprises?)\s+per\b",
        flags=re.IGNORECASE,
    ),
]

MEASURE_KEYWORDS = {
    "rate",
    "ratio",
    "average",
    "mean",
    "median",
    "total",
    "number of",
    "share of",
    "percentage of",
    "population",
    "employment",
    "unemployment",
    "production",
    "consumption",
    "income",
    "expenditure",
    "price",
    "cost",
    "mortality",
    "fertility",
    "birth",
    "death",
    "emissions",
    "energy consumption",
    "gross domestic product",
    "gdp",
    "jobs created",
    "turnover",
    "value added",
    "literacy rate",
    "recycling rate",
}


def normalize_context(value: object) -> str:
    text = unidecode(str(value))
    text = text.casefold()
    text = text.replace("\\time", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_pipe_values(value: object) -> set[str]:
    if pd.isna(value):
        return set()

    return {
        part.strip()
        for part in str(value).split("|")
        if part.strip()
    }


def column_is_unit(column_name: str) -> bool:
    normalized = normalize_context(column_name)

    if normalized in UNIT_COLUMN_NAMES:
        return True

    return normalized.endswith(" unit")


def column_is_measure(column_name: str) -> bool:
    normalized = normalize_context(column_name)

    if any(
        expression in normalized
        for expression in NON_MEASURE_COLUMN_EXPRESSIONS
    ):
        return False

    if normalized in MEASURE_COLUMN_NAMES:
        return True

    if normalized.endswith(" indicator"):
        return True

    return False


def looks_like_unit(term: str) -> bool:
    normalized = normalize_context(term)

    if any(
        pattern.search(normalized)
        for pattern in UNIT_PATTERNS
    ):
        return True

    if re.search(
        r"\b(?:kg|km|cm|mm|m2|m3|kw|mw|gw)\b",
        normalized,
    ):
        return True

    if re.search(
        r"\bindex\b.*\b100\b",
        normalized,
    ):
        return True

    return False


def looks_like_measure(term: str) -> bool:
    normalized = normalize_context(term)

    return any(
        keyword in normalized
        for keyword in MEASURE_KEYWORDS
    )


def looks_like_dimension_value(term: str) -> bool:
    normalized = normalize_context(term)

    geographical_aggregate_prefixes = (
        "euro area",
        "extra euro area",
        "non euro area",
        "european union",
        "eu countries",
    )

    if normalized.startswith(
        geographical_aggregate_prefixes
    ):
        return True

    category_prefixes = (
        "less than ",
        "more than ",
        "at least ",
        "at most ",
        "up to ",
        "under ",
        "over ",
        "between ",
        "from ",
        "below ",
        "above ",
    )

    if normalized.startswith(category_prefixes):
        return True

    return False


def is_invalid_term(term: str) -> bool:
    normalized = normalize_context(term)

    if not normalized:
        return True

    return re.search(r"[a-z0-9]", normalized) is None


def collect_term_evidence(
    table_vocabulary: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    evidence: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "term": "",
            "filenames": set(),
            "header_tables": set(),
            "title_tables": set(),
            "cell_tables": set(),
            "unit_context_tables": set(),
            "measure_context_tables": set(),
            "generic_context_tables": set(),
            "unit_only_tables": set(),
            "measure_only_tables": set(),
            "generic_only_tables": set(),
            "mixed_context_tables": set(),
            "columns": set(),
            "sources": set(),
            "total_occurrences": 0,
        }
    )

    for row in table_vocabulary.itertuples(index=False):
        normalized_term = str(row.normalized_term)
        filename = str(row.filename)
        term = str(row.term)

        sources = split_pipe_values(row.source)
        columns = split_pipe_values(row.columns)

        record = evidence[normalized_term]

        if not record["term"]:
            record["term"] = term

        record["filenames"].add(filename)
        record["columns"].update(columns)
        record["sources"].update(sources)

        try:
            occurrence_count = int(row.occurrence_count)
        except (TypeError, ValueError):
            occurrence_count = 1

        record["total_occurrences"] += occurrence_count

        if "header" in sources:
            record["header_tables"].add(filename)

        if "title" in sources:
            record["title_tables"].add(filename)

        if "cell" not in sources:
            continue

        record["cell_tables"].add(filename)

        data_columns = {
            column
            for column in columns
            if normalize_context(column) != "title"
        }

        unit_columns = {
            column
            for column in data_columns
            if column_is_unit(column)
        }

        measure_columns = {
            column
            for column in data_columns
            if column_is_measure(column)
        }

        generic_columns = {
            column
            for column in data_columns
            if column not in unit_columns
            and column not in measure_columns
        }

        if unit_columns:
            record["unit_context_tables"].add(filename)

        if measure_columns:
            record["measure_context_tables"].add(filename)

        if generic_columns:
            record["generic_context_tables"].add(filename)

        active_contexts = sum(
            [
                bool(unit_columns),
                bool(measure_columns),
                bool(generic_columns),
            ]
        )

        if active_contexts > 1:
            record["mixed_context_tables"].add(filename)
        elif unit_columns:
            record["unit_only_tables"].add(filename)
        elif measure_columns:
            record["measure_only_tables"].add(filename)
        elif generic_columns:
            record["generic_only_tables"].add(filename)

    return evidence


def classify_one_term(
    normalized_term: str,
    evidence: dict[str, object],
) -> dict[str, object]:
    term = str(evidence["term"])

    header_count = len(evidence["header_tables"])
    title_count = len(evidence["title_tables"])
    cell_count = len(evidence["cell_tables"])

    unit_context_count = len(
        evidence["unit_context_tables"]
    )

    measure_context_count = len(
        evidence["measure_context_tables"]
    )

    generic_context_count = len(
        evidence["generic_context_tables"]
    )

    unit_only_count = len(
        evidence["unit_only_tables"]
    )

    measure_only_count = len(
        evidence["measure_only_tables"]
    )

    generic_only_count = len(
        evidence["generic_only_tables"]
    )

    mixed_context_count = len(
        evidence["mixed_context_tables"]
    )

    unit_lexical = looks_like_unit(term)
    measure_lexical = looks_like_measure(term)
    dimension_value_lexical = looks_like_dimension_value(term)

    dominant_non_header = max(
        unit_only_count,
        measure_only_count,
        generic_only_count,
    )

    if is_invalid_term(term):
        category = "other"
        confidence = "low"
        reason = "The term contains no usable alphanumeric content."

    elif (
        header_count > 0
        and header_count >= dominant_non_header
        and not (
            unit_lexical
            and unit_only_count > 0
        )
        and not (
            measure_lexical
            and (
                measure_only_count > 0
                or title_count > 0
            )
        )
    ):
        category = "dimension_name"

        if mixed_context_count > 0:
            confidence = "medium"
            reason = (
                "Appears as a table header and also in mixed "
                "cell contexts."
            )
        else:
            confidence = "high"
            reason = (
                "Predominantly appears as a descriptive table header."
            )

    elif dimension_value_lexical:
        category = "dimension_value"
        confidence = "high"
        reason = (
            "Represents a quantitative category, interval or "
            "geographical aggregate rather than a measurement unit."
        )

    elif (
        measure_lexical
        and (
            measure_context_count > 0
            or title_count > 0
        )
    ):
        category = "measure"

        if measure_only_count > 0:
            confidence = "high"
            reason = (
                "Matches a measure pattern and appears in a "
                "statistical-indicator column."
            )
        else:
            confidence = "medium"
            reason = (
                "Matches a measure pattern and appears in a title "
                "or mixed statistical context."
            )

    elif (
        unit_lexical
        and (
            unit_context_count > 0
            or cell_count > 0
        )
    ):
        category = "unit"

        if unit_only_count > 0:
            confidence = "high"
            reason = (
                "Matches a unit pattern and appears exclusively "
                "in unit columns."
            )
        else:
            confidence = "medium"
            reason = (
                "Matches a unit pattern but appears in a mixed context."
            )

    elif unit_only_count > max(
        measure_only_count,
        generic_only_count,
        mixed_context_count,
    ):
        category = "unit"
        confidence = "high"
        reason = (
            "Predominantly appears exclusively in unit columns."
        )

    elif measure_only_count > max(
        unit_only_count,
        generic_only_count,
        mixed_context_count,
    ):
        category = "measure"
        confidence = "high"
        reason = (
            "Predominantly appears exclusively in statistical "
            "measure columns."
        )

    elif generic_only_count > 0 or mixed_context_count > 0:
        category = "dimension_value"

        if generic_only_count > 0:
            confidence = "high"
            reason = (
                "Appears in descriptive columns and has no "
                "dominant unit or measure evidence."
            )
        else:
            confidence = "medium"
            reason = (
                "Appears in mixed columns without a reliable "
                "unit or measure pattern."
            )

    elif title_count > 0:
        category = "measure"
        confidence = "medium"
        reason = (
            "Appears only in residual statistical table titles."
        )

    elif header_count > 0:
        category = "dimension_name"
        confidence = "medium"
        reason = "Appears as a table header."

    elif cell_count > 0:
        category = "dimension_value"
        confidence = "medium"
        reason = "Appears as a cell value without stronger evidence."

    else:
        category = "other"
        confidence = "low"
        reason = "No reliable structural context was available."

    return {
        "term": term,
        "normalized_term": normalized_term,
        "category": category,
        "confidence": confidence,
        "reason": reason,
        "table_count": len(evidence["filenames"]),
        "total_occurrences": evidence["total_occurrences"],
        "header_table_count": header_count,
        "title_table_count": title_count,
        "cell_table_count": cell_count,
        "unit_context_table_count": unit_context_count,
        "measure_context_table_count": measure_context_count,
        "generic_context_table_count": generic_context_count,
        "unit_only_table_count": unit_only_count,
        "measure_only_table_count": measure_only_count,
        "generic_only_table_count": generic_only_count,
        "mixed_context_table_count": mixed_context_count,
        "unit_lexical_match": unit_lexical,
        "measure_lexical_match": measure_lexical,
        "sources": "|".join(
            sorted(evidence["sources"])
        ),
        "columns": "|".join(
            sorted(evidence["columns"])
        ),
    }


def classify_global_vocabulary(
    table_vocabulary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evidence = collect_term_evidence(
        table_vocabulary
    )

    records = [
        classify_one_term(
            normalized_term=normalized_term,
            evidence=term_evidence,
        )
        for normalized_term, term_evidence
        in evidence.items()
    ]

    classification = pd.DataFrame(records)

    classification = classification.sort_values(
        by=[
            "category",
            "table_count",
            "total_occurrences",
            "normalized_term",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    summary = (
        classification
        .groupby(
            "category",
            as_index=False,
        )
        .agg(
            term_count=("normalized_term", "count"),
            high_confidence_count=(
                "confidence",
                lambda values: int(
                    (values == "high").sum()
                ),
            ),
            medium_confidence_count=(
                "confidence",
                lambda values: int(
                    (values == "medium").sum()
                ),
            ),
            low_confidence_count=(
                "confidence",
                lambda values: int(
                    (values == "low").sum()
                ),
            ),
        )
    )

    return classification, summary


def save_classification_results(
    classification: pd.DataFrame,
    summary: pd.DataFrame,
    output_directory: Path,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    category_files = {
        "measure": "measures.csv",
        "dimension_name": "dimension_names.csv",
        "dimension_value": "dimension_values.csv",
        "unit": "units.csv",
        "other": "other.csv",
    }

    classification.to_csv(
        output_directory / "classification_all.csv",
        index=False,
        encoding="utf-8",
    )

    summary.to_csv(
        output_directory / "classification_summary.csv",
        index=False,
        encoding="utf-8",
    )

    for category, filename in category_files.items():
        subset = classification[
            classification["category"] == category
        ].copy()

        subset.to_csv(
            output_directory / filename,
            index=False,
            encoding="utf-8",
        )

    print(
        f"Classification results saved to: {output_directory}"
    )