from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT


CATEGORIES = {
    "1": "measure",
    "2": "dimension_name",
    "3": "dimension_value",
    "4": "unit",
    "5": "other",
}


def save_progress(
    sample: pd.DataFrame,
    sample_path: Path,
) -> None:
    sample.to_csv(
        sample_path,
        index=False,
        encoding="utf-8-sig",
    )


def update_correctness(
    sample: pd.DataFrame,
    row_index: int,
) -> None:
    predicted = str(
        sample.at[row_index, "predicted_category"]
    ).strip()

    gold = str(
        sample.at[row_index, "gold_category"]
    ).strip()

    if not gold:
        sample.at[row_index, "is_correct"] = ""
    elif predicted == gold:
        sample.at[row_index, "is_correct"] = "yes"
    else:
        sample.at[row_index, "is_correct"] = "no"


def print_instructions() -> None:
    print("Manual classification annotation")
    print()
    print("Categories:")
    print("1 = measure")
    print("2 = dimension_name")
    print("3 = dimension_value")
    print("4 = unit")
    print("5 = other")
    print()
    print("Controls:")
    print("Enter = accept the predicted category")
    print("p = return to the previous row")
    print("q = save and exit")


def main() -> None:
    sample_path = (
        PROJECT_ROOT
        / "evaluation"
        / "gold_sample.csv"
    )

    if not sample_path.exists():
        raise FileNotFoundError(
            "evaluation/gold_sample.csv was not found. "
            "Run python prepare_classification_evaluation.py first."
        )

    sample = pd.read_csv(
        sample_path,
        keep_default_na=False,
    )

    required_columns = {
        "sample_id",
        "term",
        "predicted_category",
        "confidence",
        "reason",
        "columns",
        "gold_category",
        "is_correct",
        "notes",
    }

    missing_columns = required_columns.difference(
        sample.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing columns in gold_sample.csv: "
            + ", ".join(sorted(missing_columns))
        )

    print_instructions()

    annotated_mask = (
        sample["gold_category"]
        .astype(str)
        .str.strip()
        .ne("")
    )

    annotated_count = int(
        annotated_mask.sum()
    )

    print(
        f"\nExisting progress: "
        f"{annotated_count}/{len(sample)}"
    )

    if annotated_count == len(sample):
        print("The sample is already fully annotated.")
        return

    unannotated_indices = sample.index[
        ~annotated_mask
    ].tolist()

    current_position = unannotated_indices[0]

    while 0 <= current_position < len(sample):
        row = sample.loc[current_position]

        term = str(row["term"]).strip()

        if not term:
            term = "[EMPTY TERM]"

        print(
            f"\nSample {row['sample_id']} "
            f"of {len(sample)}"
        )
        print(f"Term: {term}")
        print(
            "Predicted category: "
            f"{row['predicted_category']}"
        )
        print(f"Confidence: {row['confidence']}")
        print(f"Reason: {row['reason']}")
        print(f"Columns: {row['columns']}")

        existing_gold = str(
            row["gold_category"]
        ).strip()

        if existing_gold:
            print(
                "Current gold category: "
                f"{existing_gold}"
            )

        answer = input(
            "Category [1-5, Enter, p, q]: "
        ).strip().casefold()

        if answer == "q":
            save_progress(
                sample=sample,
                sample_path=sample_path,
            )

            completed = int(
                sample["gold_category"]
                .astype(str)
                .str.strip()
                .ne("")
                .sum()
            )

            print(
                f"Progress saved: "
                f"{completed}/{len(sample)}"
            )
            return

        if answer == "p":
            if current_position > 0:
                current_position -= 1
            else:
                print("This is the first row.")

            continue

        if answer == "":
            selected_category = str(
                row["predicted_category"]
            ).strip()

        elif answer in CATEGORIES:
            selected_category = CATEGORIES[answer]

        else:
            print(
                "Invalid option. Enter 1, 2, 3, 4, 5, "
                "p, q, or press Enter."
            )
            continue

        sample.at[
            current_position,
            "gold_category",
        ] = selected_category

        update_correctness(
            sample=sample,
            row_index=current_position,
        )

        save_progress(
            sample=sample,
            sample_path=sample_path,
        )

        current_position += 1

    completed = int(
        sample["gold_category"]
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    print(
        f"Annotation completed: "
        f"{completed}/{len(sample)}"
    )
    print(f"Sample saved to: {sample_path}")


if __name__ == "__main__":
    main()