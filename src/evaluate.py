from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


VALID_CATEGORIES = [
    "measure",
    "dimension_name",
    "dimension_value",
    "unit",
    "other",
]

CONFIDENCE_ORDER = [
    "low",
    "medium",
    "high",
]


def create_stratified_gold_sample(
    classification: pd.DataFrame,
    samples_per_category: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    required_columns = {
        "term",
        "normalized_term",
        "category",
        "confidence",
        "reason",
        "table_count",
        "total_occurrences",
        "sources",
        "columns",
    }

    missing_columns = required_columns.difference(
        classification.columns
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Columns are missing in classification_all.csv: "
            f"{missing_text}"
        )

    sampled_parts: list[pd.DataFrame] = []

    for category in VALID_CATEGORIES:
        category_data = classification[
            classification["category"] == category
        ].copy()

        if category_data.empty:
            continue

        target_size = min(
            samples_per_category,
            len(category_data),
        )

        selected_indices: set[int] = set()

        confidence_target = max(
            1,
            target_size // len(CONFIDENCE_ORDER),
        )

        for confidence in CONFIDENCE_ORDER:
            confidence_data = category_data[
                category_data["confidence"] == confidence
            ]

            available_data = confidence_data[
                ~confidence_data.index.isin(
                    selected_indices
                )
            ]

            number_to_sample = min(
                confidence_target,
                len(available_data),
            )

            if number_to_sample == 0:
                continue

            sampled = available_data.sample(
                n=number_to_sample,
                random_state=(
                    random_state
                    + len(sampled_parts)
                    + len(selected_indices)
                ),
            )

            selected_indices.update(
                sampled.index.tolist()
            )

        remaining_needed = (
            target_size - len(selected_indices)
        )

        if remaining_needed > 0:
            remaining_data = category_data[
                ~category_data.index.isin(
                    selected_indices
                )
            ]

            number_to_sample = min(
                remaining_needed,
                len(remaining_data),
            )

            if number_to_sample > 0:
                sampled = remaining_data.sample(
                    n=number_to_sample,
                    random_state=(
                        random_state
                        + len(sampled_parts)
                        + 100
                    ),
                )

                selected_indices.update(
                    sampled.index.tolist()
                )

        category_sample = category_data.loc[
            sorted(selected_indices)
        ].copy()

        sampled_parts.append(category_sample)

    if not sampled_parts:
        raise ValueError(
            "No evaluation sample could be created."
        )

    sample = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    sample = sample.rename(
        columns={
            "category": "predicted_category",
        }
    )

    sample = sample[
        [
            "term",
            "normalized_term",
            "predicted_category",
            "confidence",
            "reason",
            "table_count",
            "total_occurrences",
            "sources",
            "columns",
        ]
    ].copy()

    sample.insert(
        0,
        "sample_id",
        range(1, len(sample) + 1),
    )

    sample["gold_category"] = ""
    sample["is_correct"] = ""
    sample["notes"] = ""

    return sample


def save_gold_sample(
    sample: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Muestra de evaluación guardada en: "
        f"{output_path}"
    )


def print_sample_summary(
    sample: pd.DataFrame,
) -> None:
    """Muestra el reparto de la muestra por categoría."""
    print()
    print("Distribución de la muestra:")

    counts = (
        sample["predicted_category"]
        .value_counts()
    )

    for category in VALID_CATEGORIES:
        print(
            f"  {category}: "
            f"{int(counts.get(category, 0))}"
        )

    print(
        f"  Total: {len(sample)}"
    )


def print_category_examples(
    sample: pd.DataFrame,
    examples_per_category: int = 5,
) -> None:
    """Imprime algunos ejemplos para una primera inspección."""
    for category in VALID_CATEGORIES:
        category_sample = sample[
            sample["predicted_category"] == category
        ]

        print()
        print("=" * 80)
        print(
            f"EJEMPLOS PREDICHOS COMO: {category}"
        )
        print("=" * 80)

        if category_sample.empty:
            print("[Sin ejemplos]")
            continue

        columns = [
            "term",
            "confidence",
            "reason",
            "columns",
        ]

        print(
            category_sample[
                columns
            ]
            .head(examples_per_category)
            .to_string(index=False)
        )


def evaluate_annotated_sample(
    sample: pd.DataFrame,
) -> tuple[
    dict[str, float | int],
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Evalúa la muestra después de rellenar gold_category.

    Devuelve:
    - métricas generales;
    - informe por categoría;
    - matriz de confusión.
    """
    evaluated = sample.copy()

    evaluated["gold_category"] = (
        evaluated["gold_category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    evaluated["predicted_category"] = (
        evaluated["predicted_category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    evaluated = evaluated[
        evaluated["gold_category"].isin(
            VALID_CATEGORIES
        )
    ].copy()

    if evaluated.empty:
        raise ValueError(
            "No hay filas anotadas en gold_category."
        )

    y_true = evaluated["gold_category"]
    y_pred = evaluated["predicted_category"]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    metrics = {
        "evaluated_terms": len(evaluated),
        "correct_terms": int(
            (y_true == y_pred).sum()
        ),
        "incorrect_terms": int(
            (y_true != y_pred).sum()
        ),
        "accuracy": float(accuracy),
    }

    report_dictionary = classification_report(
        y_true,
        y_pred,
        labels=VALID_CATEGORIES,
        output_dict=True,
        zero_division=0,
    )

    report_rows = []

    for category in VALID_CATEGORIES:
        category_metrics = report_dictionary[
            category
        ]

        report_rows.append(
            {
                "category": category,
                "precision": category_metrics[
                    "precision"
                ],
                "recall": category_metrics[
                    "recall"
                ],
                "f1_score": category_metrics[
                    "f1-score"
                ],
                "support": int(
                    category_metrics["support"]
                ),
            }
        )

    report = pd.DataFrame(report_rows)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=VALID_CATEGORIES,
    )

    confusion = pd.DataFrame(
        matrix,
        index=[
            f"true_{category}"
            for category in VALID_CATEGORIES
        ],
        columns=[
            f"predicted_{category}"
            for category in VALID_CATEGORIES
        ],
    )

    return metrics, report, confusion


def save_evaluation_results(
    metrics: dict[str, float | int],
    report: pd.DataFrame,
    confusion: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Guarda los resultados cuantitativos de la evaluación."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        output_directory
        / "classification_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    report.to_csv(
        output_directory
        / "classification_report.csv",
        index=False,
        encoding="utf-8",
    )

    confusion.to_csv(
        output_directory
        / "classification_confusion_matrix.csv",
        encoding="utf-8",
    )