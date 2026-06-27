import pandas as pd

from src.config import (
    PROJECT_ROOT,
    get_project_paths,
    load_config,
)
from src.evaluate import (
    create_stratified_gold_sample,
    print_category_examples,
    print_sample_summary,
    save_gold_sample,
)


def main() -> None:
    config = load_config()
    paths = get_project_paths(config)

    classification_path = (
        paths.outputs
        / "classification_all.csv"
    )

    if not classification_path.exists():
        raise FileNotFoundError(
            f"Classification file not found: {classification_path}. "
            "Run python main.py first."
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