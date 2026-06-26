from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.evaluate import (
    create_stratified_gold_sample,
    print_category_examples,
    print_sample_summary,
    save_gold_sample,
)


def main() -> None:
    classification_path = (
        PROJECT_ROOT
        / "outputs"
        / "classification_all.csv"
    )

    if not classification_path.exists():
        raise FileNotFoundError(
            "No se encuentra classification_all.csv. "
            "Ejecuta primero python main.py."
        )

    classification = pd.read_csv(
        classification_path,
        keep_default_na=False,
    )

    sample = create_stratified_gold_sample(
        classification=classification,
        samples_per_category=20,
        random_state=42,
    )

    output_path = (
        PROJECT_ROOT
        / "evaluation"
        / "gold_sample.csv"
    )

    save_gold_sample(
        sample=sample,
        output_path=output_path,
    )

    print_sample_summary(sample)

    print_category_examples(
        sample=sample,
        examples_per_category=8,
    )


if __name__ == "__main__":
    main()