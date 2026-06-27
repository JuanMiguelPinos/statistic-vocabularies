import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode


DOMAIN_PROFILES = {
    "Demography and population": {
        "description": (
            "Population structure, population change, births, deaths, "
            "fertility, migration, life expectancy and demographic trends."
        ),
        "keywords": [
            "population",
            "demographic",
            "birth",
            "death",
            "fertility",
            "mortality",
            "migration",
            "immigration",
            "emigration",
            "life expectancy",
            "population growth",
            "ageing",
            "marriage",
            "divorce",
        ],
        "strong_keywords": [
            "fertility rate",
            "birth rate",
            "death rate",
            "life expectancy",
            "population growth",
        ],
    },
    "Labour market": {
        "description": (
            "Employment, unemployment, wages, earnings, labour costs, "
            "working time, occupations and labour participation."
        ),
        "keywords": [
            "employment",
            "unemployment",
            "labour",
            "labor",
            "employee",
            "employed",
            "worker",
            "job",
            "occupation",
            "earnings",
            "wage",
            "salary",
            "working time",
            "labour cost",
            "vacancy",
            "workforce",
        ],
        "strong_keywords": [
            "unemployment rate",
            "employment rate",
            "labour force",
            "labour cost",
            "monthly earnings",
        ],
    },
    "Economy and national accounts": {
        "description": (
            "Gross domestic product, national accounts, economic growth, "
            "income, expenditure, inflation, prices and macroeconomics."
        ),
        "keywords": [
            "gdp",
            "gross domestic product",
            "national accounts",
            "economic growth",
            "inflation",
            "price index",
            "consumer price",
            "income",
            "expenditure",
            "investment",
            "value added",
            "productivity",
            "purchasing power",
            "economic activity",
        ],
        "strong_keywords": [
            "gross domestic product",
            "gdp",
            "national accounts",
            "inflation rate",
            "consumer price index",
        ],
    },
    "Business, industry and trade": {
        "description": (
            "Enterprises, industrial activity, production, turnover, "
            "business demography, international trade, imports and exports."
        ),
        "keywords": [
            "enterprise",
            "business",
            "industry",
            "industrial",
            "manufacturing",
            "production",
            "turnover",
            "company",
            "firm",
            "import",
            "export",
            "trade",
            "market share",
            "structural business",
            "retail",
            "wholesale",
        ],
        "strong_keywords": [
            "turnover",
            "industrial production",
            "business demography",
            "international trade",
            "imports",
            "exports",
        ],
    },
    "Agriculture, forestry and fisheries": {
        "description": (
            "Agricultural production, farms, livestock, crops, forestry, "
            "fishing, food production and rural activity."
        ),
        "keywords": [
            "agriculture",
            "agricultural",
            "farm",
            "farmer",
            "livestock",
            "crop",
            "cereal",
            "milk",
            "animal",
            "forest",
            "forestry",
            "fishery",
            "fisheries",
            "fishing",
            "harvest",
            "land use",
            "rural",
        ],
        "strong_keywords": [
            "agricultural production",
            "livestock",
            "crop production",
            "forestry",
            "fisheries",
        ],
    },
    "Environment and climate": {
        "description": (
            "Environmental indicators, pollution, waste, emissions, "
            "climate change, biodiversity, water and environmental quality."
        ),
        "keywords": [
            "environment",
            "environmental",
            "climate",
            "emission",
            "pollution",
            "pollutant",
            "waste",
            "recycling",
            "biodiversity",
            "greenhouse gas",
            "air quality",
            "water quality",
            "ozone",
            "particulate",
            "ecosystem",
            "carbon",
        ],
        "strong_keywords": [
            "greenhouse gas",
            "carbon emissions",
            "recycling rate",
            "air pollution",
            "climate change",
        ],
    },
    "Energy": {
        "description": (
            "Energy production, energy consumption, electricity, fuels, "
            "renewable energy, energy prices and energy efficiency."
        ),
        "keywords": [
            "energy",
            "electricity",
            "fuel",
            "gas",
            "oil",
            "coal",
            "renewable",
            "power generation",
            "energy consumption",
            "energy production",
            "energy efficiency",
            "electric power",
        ],
        "strong_keywords": [
            "energy consumption",
            "energy production",
            "renewable energy",
            "electricity generation",
            "energy efficiency",
        ],
    },
    "Transport and mobility": {
        "description": (
            "Road, rail, air, maritime and inland transport, passengers, "
            "freight, vehicles, accidents, ports and mobility."
        ),
        "keywords": [
            "transport",
            "passenger",
            "freight",
            "rail",
            "railway",
            "road",
            "vehicle",
            "car",
            "air transport",
            "airport",
            "port",
            "maritime",
            "shipping",
            "traffic",
            "mobility",
            "transport accident",
        ],
        "strong_keywords": [
            "air transport",
            "rail transport",
            "road transport",
            "passenger transport",
            "freight transport",
        ],
    },
    "Health": {
        "description": (
            "Health status, diseases, hospitals, medical services, "
            "mortality, healthcare expenditure and public health."
        ),
        "keywords": [
            "health",
            "disease",
            "hospital",
            "patient",
            "doctor",
            "physician",
            "nurse",
            "medical",
            "healthcare",
            "health care",
            "disability",
            "cancer",
            "infection",
            "cause of death",
            "bed place",
        ],
        "strong_keywords": [
            "health expenditure",
            "hospital beds",
            "cause of death",
            "healthcare",
            "medical treatment",
        ],
    },
    "Education and training": {
        "description": (
            "Education participation, educational attainment, schools, "
            "students, graduates, literacy and vocational training."
        ),
        "keywords": [
            "education",
            "educational",
            "school",
            "student",
            "graduate",
            "teacher",
            "training",
            "learning",
            "literacy",
            "school completion",
            "tertiary education",
            "vocational training",
            "isced",
        ],
        "strong_keywords": [
            "educational attainment",
            "school completion",
            "literacy rate",
            "tertiary education",
            "vocational training",
        ],
    },
    "Science, technology and digital society": {
        "description": (
            "Research, innovation, science, technology, digital skills, "
            "internet use, information society and research expenditure."
        ),
        "keywords": [
            "research",
            "researcher",
            "science",
            "technology",
            "innovation",
            "digital",
            "internet",
            "ict",
            "information society",
            "r d",
            "research and development",
            "patent",
            "broadband",
            "computer",
            "online",
        ],
        "strong_keywords": [
            "research and development",
            "digital skills",
            "internet use",
            "information society",
            "research expenditure",
        ],
    },
    "Income, poverty and living conditions": {
        "description": (
            "Poverty, social exclusion, household income, inequality, "
            "material deprivation and quality of life."
        ),
        "keywords": [
            "poverty",
            "at risk of poverty",
            "social exclusion",
            "income inequality",
            "household income",
            "material deprivation",
            "living conditions",
            "quality of life",
            "low income",
            "jobless household",
            "social transfer",
        ],
        "strong_keywords": [
            "at risk of poverty",
            "poverty rate",
            "material deprivation",
            "income inequality",
            "social exclusion",
        ],
    },
    "Government and public finance": {
        "description": (
            "Government expenditure, public revenue, public debt, taxes, "
            "deficit, public administration and government accounts."
        ),
        "keywords": [
            "government",
            "public expenditure",
            "public revenue",
            "public debt",
            "deficit",
            "tax",
            "taxation",
            "public finance",
            "government expenditure",
            "government revenue",
            "public administration",
            "budget",
        ],
        "strong_keywords": [
            "government expenditure",
            "public debt",
            "government deficit",
            "tax revenue",
            "public finance",
        ],
    },
    "Crime, justice and safety": {
        "description": (
            "Crime, violence, police, justice, courts, prisons, accidents "
            "and personal or public safety."
        ),
        "keywords": [
            "crime",
            "criminal",
            "violence",
            "police",
            "court",
            "justice",
            "prison",
            "offence",
            "victim",
            "perpetrator",
            "safety",
            "accident",
            "injury",
            "homicide",
        ],
        "strong_keywords": [
            "crime rate",
            "domestic violence",
            "law courts",
            "road accidents",
            "criminal offence",
        ],
    },
    "Tourism": {
        "description": (
            "Tourist arrivals, accommodation, nights spent, hotels, "
            "tourism capacity and tourism activity."
        ),
        "keywords": [
            "tourism",
            "tourist",
            "hotel",
            "accommodation",
            "nights spent",
            "arrival",
            "bed place",
            "tourism establishment",
            "holiday",
            "travel accommodation",
        ],
        "strong_keywords": [
            "tourist arrivals",
            "tourism accommodation",
            "nights spent",
            "bed places",
            "tourism establishments",
        ],
    },
    "Housing and construction": {
        "description": (
            "Housing, dwellings, buildings, construction activity, "
            "house prices, rents and residential conditions."
        ),
        "keywords": [
            "housing",
            "dwelling",
            "building",
            "construction",
            "house price",
            "rent",
            "residential",
            "home ownership",
            "building permit",
            "housing cost",
        ],
        "strong_keywords": [
            "house prices",
            "housing costs",
            "building permits",
            "construction activity",
            "residential buildings",
        ],
    },
}


OTHER_DOMAIN = "Other or multidisciplinary"

DOMAIN_OVERRIDE_RULES = [
    (
        "Crime, justice and safety",
        [
            ("violence", r"\bviolence\b"),
            ("homicide", r"\bhomicide\b"),
            ("crime", r"\bcrime\b"),
            ("criminal offence", r"\bcriminal offences?\b"),
            ("victim", r"\bvictims?\b"),
            ("perpetrator", r"\bperpetrators?\b"),
            ("law court", r"\blaw courts?\b"),
            ("prison", r"\bprisons?\b"),
        ],
    ),
    (
        "Health",
        [
            ("health", r"\bhealth\b"),
            ("health care", r"\bhealth care\b"),
            ("healthcare", r"\bhealthcare\b"),
            ("disease", r"\bdiseases?\b"),
            ("hospital", r"\bhospitals?\b"),
            ("medical doctor", r"\bmedical doctors?\b"),
            ("antibiotic", r"\bantibiotics?\b"),
            ("diagnosis", r"\bdiagnosis\b"),
            ("cause of death", r"\bcause of death\b"),
            ("death due to disease", r"\bdeath due to\b"),
            ("health problem", r"\bhealth problems?\b"),
        ],
    ),
    (
        "Government and public finance",
        [
            ("government", r"\bgovernment\b"),
            ("public debt", r"\bpublic debt\b"),
            ("government debt", r"\bgovernment debt\b"),
            ("deficit", r"\bdeficit\b"),
            ("surplus", r"\bsurplus\b"),
            ("bond yield", r"\bbond yields?\b"),
            ("public expenditure", r"\bpublic expenditure\b"),
            ("public revenue", r"\bpublic revenue\b"),
            ("tax", r"\btax(?:es|ation)?\b"),
            ("budget", r"\bbudget\b"),
        ],
    ),
    (
        "Income, poverty and living conditions",
        [
            ("poverty", r"\bpoverty\b"),
            ("deprivation", r"\bdepriv\w*\b"),
            ("income distribution", r"\bincome distribution\b"),
            ("income inequality", r"\bincome inequality\b"),
            ("living conditions", r"\bliving conditions\b"),
            ("social exclusion", r"\bsocial exclusion\b"),
            ("jobless household", r"\bjobless households?\b"),
            ("housing cost burden", r"\bhousing cost burden\b"),
        ],
    ),
    (
        "Transport and mobility",
        [
            ("air transport", r"\bair transport\b"),
            ("road transport", r"\broad transport\b"),
            ("goods transport", r"\bgoods transport\b"),
            ("passenger car", r"\bpassenger cars?\b"),
            ("lorry", r"\blorr(?:y|ies)\b"),
            ("trailer", r"\btrailers?\b"),
            ("railway", r"\brailways?\b"),
            ("rail transport", r"\brail transport\b"),
            ("rolling stock", r"\brolling stock\b"),
            ("maritime", r"\bmaritime\b"),
            ("ship", r"\bships?\b"),
            ("road accident", r"\broad accidents?\b"),
            ("port", r"\bports?\b"),
            ("freight vessel", r"\bfreight vessels?\b"),
        ],
    ),
    (
        "Environment and climate",
        [
            ("greenhouse gas", r"\bgreenhouse gases?\b"),
            ("emission", r"\bemissions?\b"),
            ("pollution", r"\bpollution\b"),
            ("waste", r"\bwaste\b"),
            ("sanitation", r"\bsanitation\b"),
            ("material consumption", r"\bmaterial consumption\b"),
            ("climate", r"\bclimate\b"),
            ("recycling", r"\brecycling\b"),
            ("ozone", r"\bozone\b"),
            ("air quality", r"\bair quality\b"),
        ],
    ),
    (
        "Labour market",
        [
            ("unemployment", r"\bunemployment\b"),
            ("employment", r"\bemployment\b"),
            ("employed", r"\bemployed\b"),
            ("labour", r"\blabou?r\b"),
            ("job vacancy", r"\bjob vacancies?\b"),
            ("earnings", r"\bearnings?\b"),
            ("wage", r"\bwages?\b"),
            ("working time", r"\bworking time\b"),
            ("part-time work", r"\bpart time\b"),
        ],
    ),
    (
        "Education and training",
        [
            ("education", r"\beducation\b"),
            ("educational", r"\beducational\b"),
            ("school", r"\bschool\b"),
            ("teacher", r"\bteachers?\b"),
            ("student", r"\bstudents?\b"),
            ("ISCED", r"\bisced\b"),
            ("literacy", r"\bliteracy\b"),
            ("enrolment", r"\benrolment\b"),
            ("training", r"\btraining\b"),
        ],
    ),
    (
        "Science, technology and digital society",
        [
            ("internet", r"\binternet\b"),
            ("digital", r"\bdigital\b"),
            ("science", r"\bscience\b"),
            ("technology", r"\btechnology\b"),
            ("research", r"\bresearch\b"),
            ("innovation", r"\binnovation\b"),
            ("R&D", r"\br d\b"),
            ("email", r"\be mails?\b|\bemail\b"),
        ],
    ),
    (
        "Energy",
        [
            ("renewable energy", r"\brenewable energy\b"),
            ("energy production", r"\benergy production\b"),
            ("energy consumption", r"\benergy consumption\b"),
            ("energy efficiency", r"\benergy efficiency\b"),
            ("electricity", r"\belectricity\b"),
            ("power generation", r"\bpower generation\b"),
        ],
    ),
    (
        "Tourism",
        [
            ("tourism", r"\btouris\w*\b"),
            ("tourist", r"\btourists?\b"),
            ("nights spent", r"\bnights spent\b"),
            ("hotel", r"\bhotels?\b"),
            ("accommodation", r"\baccommodation\b"),
        ],
    ),
    (
        "Housing and construction",
        [
            ("housing", r"\bhousing\b"),
            ("dwelling", r"\bdwellings?\b"),
            ("building permit", r"\bbuilding permits?\b"),
            ("construction", r"\bconstruction\b"),
            ("house price", r"\bhouse prices?\b"),
            ("rent", r"\brents?\b"),
        ],
    ),
    (
        "Agriculture, forestry and fisheries",
        [
            ("agriculture", r"\bagricultur\w*\b"),
            ("livestock", r"\blivestock\b"),
            ("crop", r"\bcrops?\b"),
            ("forestry", r"\bforestr\w*\b"),
            ("fisheries", r"\bfisher\w*\b"),
            ("farm", r"\bfarms?\b"),
            ("harvest", r"\bharvest\w*\b"),
        ],
    ),
    (
        "Business, industry and trade",
        [
            ("turnover", r"\bturnover\b"),
            ("industry", r"\bindustry\b"),
            ("manufacturing", r"\bmanufactur\w*\b"),
            ("imports", r"\bimports?\b"),
            ("exports", r"\bexports?\b"),
            ("trade integration", r"\btrade integration\b"),
            ("market integration", r"\bmarket integration\b"),
            ("production activities", r"\bproduction activities\b"),
        ],
    ),
    (
        "Economy and national accounts",
        [
            ("GDP", r"\bgdp\b"),
            ("gross domestic product", r"\bgross domestic product\b"),
            ("gross value added", r"\bgross value added\b"),
            ("exchange rate", r"\bexchange rate\b"),
            ("consumer confidence", r"\bconsumer confidence\b"),
            ("national accounts", r"\bnational accounts\b"),
            ("inflation", r"\binflation\b"),
            ("productivity", r"\bproductivity\b"),
        ],
    ),
    (
        "Demography and population",
        [
            ("immigration", r"\bimmigration\b"),
            ("emigration", r"\bemigration\b"),
            ("fertility", r"\bfertility\b"),
            ("birth rate", r"\bbirth rate\b"),
            ("life expectancy", r"\blife expectancy\b"),
            ("demography", r"\bdemograph\w*\b"),
        ],
    ),
]


def normalize_domain_text(value: object) -> str:
    text = unidecode(str(value))
    text = text.casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def apply_domain_override(
    normalized_text: str,
) -> tuple[str | None, list[str]]:
    for domain, rules in DOMAIN_OVERRIDE_RULES:
        matches = [
            label
            for label, pattern in rules
            if re.search(pattern, normalized_text)
        ]

        if matches:
            return domain, matches

    return None, []


def keyword_is_present(
    normalized_text: str,
    keyword: str,
) -> bool:
    normalized_keyword = normalize_domain_text(keyword)

    if not normalized_keyword:
        return False

    padded_text = f" {normalized_text} "
    padded_keyword = f" {normalized_keyword} "

    return padded_keyword in padded_text


def find_domain_keywords(
    normalized_text: str,
    profile: dict[str, object],
) -> tuple[list[str], list[str]]:
    keyword_matches = [
        keyword
        for keyword in profile["keywords"]
        if keyword_is_present(
            normalized_text,
            keyword,
        )
    ]

    strong_matches = [
        keyword
        for keyword in profile["strong_keywords"]
        if keyword_is_present(
            normalized_text,
            keyword,
        )
    ]

    return keyword_matches, strong_matches


def build_domain_prototype(
    domain: str,
    profile: dict[str, object],
) -> str:
    return " ".join(
        [
            domain,
            str(profile["description"]),
            " ".join(profile["keywords"]),
            " ".join(profile["strong_keywords"]),
        ]
    )


def assign_confidence(
    score: float,
    margin: float,
    strong_match_count: int,
) -> str:
    if (
        score >= 0.55
        or (
            strong_match_count > 0
            and margin >= 0.08
        )
    ):
        return "high"

    if score >= 0.25 or margin >= 0.05:
        return "medium"

    return "low"


def cluster_measures_by_domain(
    measures: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if measures.empty:
        raise ValueError(
            "No measures were available for domain assignment."
        )

    required_columns = {
        "term",
        "normalized_term",
        "table_count",
        "total_occurrences",
    }

    missing_columns = required_columns.difference(
        measures.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing columns in measures.csv: "
            + ", ".join(sorted(missing_columns))
        )

    domain_names = list(DOMAIN_PROFILES)

    measure_texts = [
        normalize_domain_text(term)
        for term in measures["term"]
    ]

    prototype_texts = [
        normalize_domain_text(
            build_domain_prototype(
                domain=domain,
                profile=DOMAIN_PROFILES[domain],
            )
        )
        for domain in domain_names
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        measure_texts + prototype_texts
    )

    measure_matrix = matrix[
        :len(measure_texts)
    ]

    prototype_matrix = matrix[
        len(measure_texts):
    ]

    semantic_scores = cosine_similarity(
        measure_matrix,
        prototype_matrix,
    )

    records: list[dict[str, object]] = []

    for row_position, row in enumerate(
        measures.itertuples(index=False)
    ):
        term = str(row.term)
        normalized_text = measure_texts[row_position]

        combined_scores = semantic_scores[
            row_position
        ].copy()

        keyword_information: dict[
            str,
            tuple[list[str], list[str]]
        ] = {}

        for domain_position, domain in enumerate(
            domain_names
        ):
            profile = DOMAIN_PROFILES[domain]

            (
                keyword_matches,
                strong_matches,
            ) = find_domain_keywords(
                normalized_text=normalized_text,
                profile=profile,
            )

            keyword_information[domain] = (
                keyword_matches,
                strong_matches,
            )

            keyword_boost = min(
                0.30,
                0.08 * len(keyword_matches),
            )

            strong_boost = min(
                0.45,
                0.22 * len(strong_matches),
            )

            combined_scores[domain_position] += (
                keyword_boost
                + strong_boost
            )

        ranking = np.argsort(
            combined_scores
        )[::-1]

        (
            override_domain,
            override_matches,
        ) = apply_domain_override(
            normalized_text
        )

        if override_domain is not None:
            assigned_domain = override_domain
            best_domain = override_domain

            best_position = domain_names.index(
                override_domain
            )

            second_position = next(
                int(position)
                for position in ranking
                if int(position) != best_position
            )

            second_domain = domain_names[
                second_position
            ]

            best_score = float(
                combined_scores[best_position]
            )

            second_score = float(
                combined_scores[second_position]
            )

            margin = best_score - second_score

            (
                matched_keywords,
                matched_strong_keywords,
            ) = keyword_information[best_domain]

            assignment_method = (
                "rule_override_and_tfidf"
            )

            confidence = "high"

        else:
            best_position = int(ranking[0])
            second_position = int(ranking[1])

            best_domain = domain_names[
                best_position
            ]

            second_domain = domain_names[
                second_position
            ]

            best_score = float(
                combined_scores[best_position]
            )

            second_score = float(
                combined_scores[second_position]
            )

            margin = best_score - second_score

            (
                matched_keywords,
                matched_strong_keywords,
            ) = keyword_information[best_domain]

            if (
                best_score < 0.08
                and not matched_keywords
                and not matched_strong_keywords
            ):
                assigned_domain = OTHER_DOMAIN
                assignment_method = (
                    "low_similarity_fallback"
                )
                confidence = "low"
            else:
                assigned_domain = best_domain

                if matched_strong_keywords:
                    assignment_method = (
                        "tfidf_and_strong_keywords"
                    )
                elif matched_keywords:
                    assignment_method = (
                        "tfidf_and_keywords"
                    )
                else:
                    assignment_method = "tfidf"

                confidence = assign_confidence(
                    score=best_score,
                    margin=margin,
                    strong_match_count=len(
                        matched_strong_keywords
                    ),
                )

        reason_parts = [
            f"Best profile: {best_domain}",
            f"score={best_score:.4f}",
            f"margin={margin:.4f}",
        ]

        if override_matches:
            reason_parts.append(
                "override_rules="
                + ", ".join(override_matches)
            )

        if matched_keywords:
            reason_parts.append(
                "keywords="
                + ", ".join(matched_keywords)
            )

        if matched_strong_keywords:
            reason_parts.append(
                "strong_keywords="
                + ", ".join(
                    matched_strong_keywords
                )
            )

        records.append(
            {
                "term": term,
                "normalized_term": (
                    row.normalized_term
                ),
                "domain": assigned_domain,
                "domain_score": round(
                    best_score,
                    6,
                ),
                "second_domain": second_domain,
                "second_domain_score": round(
                    second_score,
                    6,
                ),
                "score_margin": round(
                    margin,
                    6,
                ),
                "confidence": confidence,
                "assignment_method": (
                    assignment_method
                ),
                "matched_keywords": "|".join(
                    matched_keywords
                ),
                "matched_strong_keywords": (
                    "|".join(
                        matched_strong_keywords
                    )
                ),
                "reason": "; ".join(
                    reason_parts
                ),
                "table_count": row.table_count,
                "total_occurrences": (
                    row.total_occurrences
                ),
                "matched_override_rules": "|".join(
                    override_matches
                ),
            }
        )

    assignments = pd.DataFrame(records)

    assignments = assignments.sort_values(
        by=[
            "domain",
            "domain_score",
            "table_count",
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
        assignments
        .groupby(
            "domain",
            as_index=False,
        )
        .agg(
            measure_count=(
                "normalized_term",
                "count",
            ),
            average_score=(
                "domain_score",
                "mean",
            ),
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

    summary["average_score"] = (
        summary["average_score"].round(4)
    )

    summary = summary.sort_values(
        by="measure_count",
        ascending=False,
    ).reset_index(drop=True)

    examples = (
        assignments
        .sort_values(
            by=[
                "domain",
                "domain_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "domain",
            as_index=False,
            group_keys=False,
        )
        .head(10)
        .reset_index(drop=True)
    )

    return assignments, summary, examples


def save_domain_results(
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    examples: pd.DataFrame,
    output_directory: Path,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignments_path = (
        output_directory
        / "measure_domains.csv"
    )

    summary_path = (
        output_directory
        / "domain_summary.csv"
    )

    examples_path = (
        output_directory
        / "domain_examples.csv"
    )

    assignments.to_csv(
        assignments_path,
        index=False,
        encoding="utf-8",
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8",
    )

    examples.to_csv(
        examples_path,
        index=False,
        encoding="utf-8",
    )

    print(f"Domain results saved to: {output_directory}")