from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT


CONFIDENCE_LEVELS = [
    "low",
    "medium",
    "high",
]


def create_domain_sample(
    assignments: pd.DataFrame,
    samples_per_domain: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Crea una muestra estratificada de medidas por dominio.

    Intenta incluir ejemplos de confianza baja, media y alta.
    """
    required_columns = {
        "term",
        "normalized_term",
        "domain",
        "domain_score",
        "second_domain",
        "second_domain_score",
        "score_margin",
        "confidence",
        "assignment_method",
        "matched_keywords",
        "reason",
    }

    missing_columns = required_columns.difference(
        assignments.columns
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas en measure_domains.csv: "
            + ", ".join(sorted(missing_columns))
        )

    sampled_parts: list[pd.DataFrame] = []

    domains = sorted(
        assignments["domain"]
        .dropna()
        .unique()
        .tolist()
    )

    for domain_index, domain in enumerate(domains):
        domain_data = assignments[
            assignments["domain"] == domain
        ].copy()

        target_size = min(
            samples_per_domain,
            len(domain_data),
        )

        selected_indices: set[int] = set()

        # Intenta seleccionar al menos un ejemplo
        # por nivel de confianza.
        for confidence_index, confidence in enumerate(
            CONFIDENCE_LEVELS
        ):
            confidence_data = domain_data[
                domain_data["confidence"] == confidence
            ]

            available_data = confidence_data[
                ~confidence_data.index.isin(
                    selected_indices
                )
            ]

            if available_data.empty:
                continue

            sampled = available_data.sample(
                n=1,
                random_state=(
                    random_state
                    + domain_index * 10
                    + confidence_index
                ),
            )

            selected_indices.update(
                sampled.index.tolist()
            )

        remaining_needed = (
            target_size - len(selected_indices)
        )

        if remaining_needed > 0:
            remaining_data = domain_data[
                ~domain_data.index.isin(
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
                        + domain_index
                        + 100
                    ),
                )

                selected_indices.update(
                    sampled.index.tolist()
                )

        domain_sample = domain_data.loc[
            sorted(selected_indices)
        ].copy()

        sampled_parts.append(domain_sample)

    sample = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    sample = sample.rename(
        columns={
            "domain": "predicted_domain",
        }
    )

    sample.insert(
        0,
        "sample_id",
        range(1, len(sample) + 1),
    )

    sample["gold_domain"] = ""
    sample["is_correct"] = ""
    sample["notes"] = ""

    selected_columns = [
        "sample_id",
        "term",
        "normalized_term",
        "predicted_domain",
        "domain_score",
        "second_domain",
        "second_domain_score",
        "score_margin",
        "confidence",
        "assignment_method",
        "matched_keywords",
        "reason",
        "gold_domain",
        "is_correct",
        "notes",
    ]

    return sample[selected_columns]


def print_sample_summary(
    sample: pd.DataFrame,
) -> None:
    """Imprime la distribución de la muestra."""
    print()
    print("Distribución de la muestra por dominio:")

    counts = (
        sample["predicted_domain"]
        .value_counts()
        .sort_index()
    )

    for domain, count in counts.items():
        print(f"  {domain}: {count}")

    print()
    print(f"Total de términos: {len(sample)}")


def print_domain_examples(
    sample: pd.DataFrame,
) -> None:
    """Imprime los términos seleccionados por dominio."""
    domains = sorted(
        sample["predicted_domain"]
        .unique()
        .tolist()
    )

    for domain in domains:
        domain_sample = sample[
            sample["predicted_domain"] == domain
        ]

        print()
        print("=" * 80)
        print(f"DOMINIO PREDICHO: {domain}")
        print("=" * 80)

        print(
            domain_sample[
                [
                    "sample_id",
                    "term",
                    "confidence",
                    "domain_score",
                    "second_domain",
                ]
            ].to_string(index=False)
        )


def main() -> None:
    assignments_path = (
        PROJECT_ROOT
        / "outputs"
        / "measure_domains.csv"
    )

    output_path = (
        PROJECT_ROOT
        / "evaluation"
        / "domain_gold_sample.csv"
    )

    if not assignments_path.exists():
        raise FileNotFoundError(
            "No se encuentra outputs/measure_domains.csv. "
            "Ejecuta primero python main.py."
        )

    assignments = pd.read_csv(
        assignments_path,
        keep_default_na=False,
    )

    sample = create_domain_sample(
        assignments=assignments,
        samples_per_domain=5,
        random_state=42,
    )

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
        "Muestra de dominios guardada en: "
        f"{output_path}"
    )

    print_sample_summary(sample)
    print_domain_examples(sample)


if __name__ == "__main__":
    main()