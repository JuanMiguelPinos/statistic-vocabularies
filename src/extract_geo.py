import re
import unicodedata
from pathlib import Path

import pandas as pd
from unidecode import unidecode

from src.extract_vocabulary import build_global_table_vocabulary
from src.io_utils import find_single_file, read_excel_sheet


COUNTRY_ALIASES = {
    "AL": ["Albania"],
    "AT": ["Austria"],
    "BA": ["Bosnia and Herzegovina"],
    "BE": ["Belgium"],
    "BG": ["Bulgaria"],
    "CH": ["Switzerland"],
    "CY": ["Cyprus"],
    "CZ": ["Czechia", "Czech Republic"],
    "DE": ["Germany"],
    "DK": ["Denmark"],
    "EE": ["Estonia"],
    "EL": ["Greece"],
    "ES": ["Spain"],
    "FI": ["Finland"],
    "FR": ["France"],
    "HR": ["Croatia"],
    "HU": ["Hungary"],
    "IE": ["Ireland"],
    "IS": ["Iceland"],
    "IT": ["Italy"],
    "LI": ["Liechtenstein"],
    "LT": ["Lithuania"],
    "LU": ["Luxembourg"],
    "LV": ["Latvia"],
    "ME": ["Montenegro"],
    "MK": ["North Macedonia"],
    "MT": ["Malta"],
    "NL": ["Netherlands"],
    "NO": ["Norway"],
    "PL": ["Poland"],
    "PT": ["Portugal"],
    "RO": ["Romania"],
    "RS": ["Serbia"],
    "SE": ["Sweden"],
    "SI": ["Slovenia"],
    "SK": ["Slovakia", "Slovak Republic"],
    "TR": ["Türkiye", "Turkey"],
    "UK": ["United Kingdom"],
    "XK": ["Kosovo", "Kosovo*"],
}


def normalize_geography(value: object) -> str:
    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = unidecode(text)
    text = text.casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def generate_label_variants(label: object) -> set[str]:
    text = str(label).strip()

    if not text:
        return set()

    variants = {text}

    for separator in ["/", ";"]:
        if separator in text:
            variants.update(
                part.strip()
                for part in text.split(separator)
                if part.strip()
            )

    return variants


def build_geography_dictionary(
    nuts_path: Path,
) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def register(
        alias: object,
        canonical_name: object,
        code: object,
        source: str,
    ) -> None:
        alias_text = str(alias).strip()
        canonical_text = str(canonical_name).strip()
        code_text = str(code).strip()

        normalized_alias = normalize_geography(alias_text)

        if not normalized_alias:
            return

        key = (normalized_alias, code_text)

        if key in seen:
            return

        seen.add(key)

        records.append(
            {
                "alias": alias_text,
                "normalized_alias": normalized_alias,
                "canonical_name": canonical_text,
                "geo_code": code_text,
                "dictionary_source": source,
            }
        )

    nuts = read_excel_sheet(
        nuts_path,
        sheet_name="NUTS2024",
    )

    for _, row in nuts.iterrows():
        code = row.get("NUTS Code", "")
        label = row.get("NUTS label", "")

        register(
            alias=code,
            canonical_name=label,
            code=code,
            source="NUTS2024_code",
        )

        for variant in generate_label_variants(label):
            register(
                alias=variant,
                canonical_name=label,
                code=code,
                source="NUTS2024_label",
            )

    statistical_regions = read_excel_sheet(
        nuts_path,
        sheet_name="Statistical Regions",
    )

    for _, row in statistical_regions.iterrows():
        code = row.get("SR Code", "")
        label = row.get("SR label", "")

        register(
            alias=code,
            canonical_name=label,
            code=code,
            source="Statistical_region_code",
        )

        for variant in generate_label_variants(label):
            register(
                alias=variant,
                canonical_name=label,
                code=code,
                source="Statistical_region_label",
            )

    transliterations = read_excel_sheet(
        nuts_path,
        sheet_name="Cyrillic & Greek to Latin",
    )

    for _, row in transliterations.iterrows():
        code = row.get("Code", "")
        original_label = row.get("Label", "")
        latin_label = row.get("Transliteration to Latin", "")

        register(
            alias=original_label,
            canonical_name=latin_label or original_label,
            code=code,
            source="Original_label",
        )

        register(
            alias=latin_label,
            canonical_name=latin_label,
            code=code,
            source="Latin_transliteration",
        )

    for country_code, aliases in COUNTRY_ALIASES.items():
        canonical_name = aliases[0]

        register(
            alias=country_code,
            canonical_name=canonical_name,
            code=country_code,
            source="Country_code",
        )

        for alias in aliases:
            register(
                alias=alias,
                canonical_name=canonical_name,
                code=country_code,
                source="Country_name",
            )

    dictionary = pd.DataFrame(
        records,
        columns=[
            "alias",
            "normalized_alias",
            "canonical_name",
            "geo_code",
            "dictionary_source",
        ],
    )

    return dictionary.sort_values(
        by=["normalized_alias", "geo_code"]
    ).reset_index(drop=True)


def identify_geographical_terms(
    table_vocabulary: pd.DataFrame,
    geography_dictionary: pd.DataFrame,
) -> pd.DataFrame:
    alias_lookup: dict[str, dict[str, str]] = {}

    for _, row in geography_dictionary.iterrows():
        normalized_alias = row["normalized_alias"]

        if normalized_alias not in alias_lookup:
            alias_lookup[normalized_alias] = {
                "canonical_name": row["canonical_name"],
                "geo_code": row["geo_code"],
                "dictionary_source": row["dictionary_source"],
            }

    geography_records: list[dict[str, object]] = []

    for _, row in table_vocabulary.iterrows():
        term = row["term"]
        normalized_geo = normalize_geography(term)

        match = alias_lookup.get(normalized_geo)

        if match is None:
            continue

        geography_records.append(
            {
                "filename": row["filename"],
                "term": term,
                "normalized_term": row["normalized_term"],
                "matched_geography": match["canonical_name"],
                "geo_code": match["geo_code"],
                "dictionary_source": match["dictionary_source"],
                "occurrence_count": row["occurrence_count"],
            }
        )

    return pd.DataFrame(
        geography_records,
        columns=[
            "filename",
            "term",
            "normalized_term",
            "matched_geography",
            "geo_code",
            "dictionary_source",
            "occurrence_count",
        ],
    )


def remove_geographies_from_vocabulary(
    table_vocabulary: pd.DataFrame,
    geographical_terms: pd.DataFrame,
) -> pd.DataFrame:
    if geographical_terms.empty:
        return table_vocabulary.copy()

    geographical_keys = set(
        zip(
            geographical_terms["filename"],
            geographical_terms["normalized_term"],
        )
    )

    keep_mask = [
        (filename, normalized_term) not in geographical_keys
        for filename, normalized_term in zip(
            table_vocabulary["filename"],
            table_vocabulary["normalized_term"],
        )
    ]

    return table_vocabulary.loc[keep_mask].reset_index(drop=True)


def build_geography_summary(
    table_vocabulary: pd.DataFrame,
    vocabulary_without_geo: pd.DataFrame,
    geographical_terms: pd.DataFrame,
) -> pd.DataFrame:
    original_counts = (
        table_vocabulary.groupby("filename")
        .size()
        .rename("s_term_count")
    )

    final_counts = (
        vocabulary_without_geo.groupby("filename")
        .size()
        .rename("v_term_count")
    )

    if geographical_terms.empty:
        geo_counts = pd.Series(
            dtype=int,
            name="geo_term_count",
        )
    else:
        geo_counts = (
            geographical_terms.groupby("filename")
            .size()
            .rename("geo_term_count")
        )

    summary = pd.concat(
        [
            original_counts,
            geo_counts,
            final_counts,
        ],
        axis=1,
    ).fillna(0)

    summary = summary.reset_index()

    count_columns = [
        "s_term_count",
        "geo_term_count",
        "v_term_count",
    ]

    summary[count_columns] = summary[count_columns].astype(int)

    return summary


def save_geography_results(
    geography_dictionary: pd.DataFrame,
    geographical_terms: pd.DataFrame,
    vocabulary_without_geo: pd.DataFrame,
    global_vocabulary_without_geo: pd.DataFrame,
    summary: pd.DataFrame,
    output_directory: Path,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    geography_dictionary.to_csv(
        output_directory / "geography_dictionary.csv",
        index=False,
        encoding="utf-8",
    )

    geographical_terms.to_csv(
        output_directory / "table_geographies.csv",
        index=False,
        encoding="utf-8",
    )

    vocabulary_without_geo.to_csv(
        output_directory / "table_vocabulary_without_geo.csv",
        index=False,
        encoding="utf-8",
    )

    global_vocabulary_without_geo.to_csv(
        output_directory / "global_vocabulary_without_geo.csv",
        index=False,
        encoding="utf-8",
    )

    summary.to_csv(
        output_directory / "geography_summary.csv",
        index=False,
        encoding="utf-8",
    )

    print(f"Geography results saved to: {output_directory}")