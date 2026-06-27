from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROJECT_ROOT
from src.evaluate import (
    VALID_CATEGORIES,
    evaluate_annotated_sample,
    save_evaluation_results,
)


def save_confusion_matrix_figure(
    confusion: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    image = axis.imshow(
        confusion.values
    )

    axis.set_xticks(
        range(len(confusion.columns))
    )

    axis.set_yticks(
        range(len(confusion.index))
    )

    axis.set_xticklabels(
        VALID_CATEGORIES,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        VALID_CATEGORIES
    )

    axis.set_xlabel(
        "Predicted category"
    )

    axis.set_ylabel(
        "Gold category"
    )

    axis.set_title(
        "Classification confusion matrix"
    )

    for row_index in range(
        confusion.shape[0]
    ):
        for column_index in range(
            confusion.shape[1]
        ):
            axis.text(
                column_index,
                row_index,
                int(
                    confusion.iloc[
                        row_index,
                        column_index,
                    ]
                ),
                ha="center",
                va="center",
            )

    figure.colorbar(
        image,
        ax=axis,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def update_sample_correctness(
    sample: pd.DataFrame,
) -> pd.DataFrame:
    updated = sample.copy()

    predicted = (
        updated["predicted_category"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    gold = (
        updated["gold_category"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    updated["is_correct"] = (
        predicted == gold
    ).map(
        {
            True: "yes",
            False: "no",
        }
    )

    return updated


def main() -> None:
    sample_path = (
        PROJECT_ROOT
        / "evaluation"
        / "gold_sample.csv"
    )

    output_directory = (
        PROJECT_ROOT
        / "evaluation"
    )

    figure_path = (
        output_directory
        / "figures"
        / "classification_confusion_matrix.png"
    )

    if not sample_path.exists():
        raise FileNotFoundError(
            "evaluation/gold_sample.csv was not found."
        )

    sample = pd.read_csv(
        sample_path,
        keep_default_na=False,
    )

    missing_annotations = sample[
        sample["gold_category"]
        .astype(str)
        .str.strip()
        .eq("")
    ]

    if not missing_annotations.empty:
        raise ValueError(
            "Unannotated terms remain: "
            f"{len(missing_annotations)}"
        )

    invalid_annotations = sample[
        ~sample["gold_category"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .isin(VALID_CATEGORIES)
    ]

    if not invalid_annotations.empty:
        invalid_values = sorted(
            invalid_annotations[
                "gold_category"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Invalid gold categories were found: "
            f"{invalid_values}"
        )

    sample = update_sample_correctness(
        sample
    )

    sample.to_csv(
        sample_path,
        index=False,
        encoding="utf-8-sig",
    )

    (
        metrics,
        report,
        confusion,
    ) = evaluate_annotated_sample(
        sample
    )

    save_evaluation_results(
        metrics=metrics,
        report=report,
        confusion=confusion,
        output_directory=output_directory,
    )

    save_confusion_matrix_figure(
        confusion=confusion,
        output_path=figure_path,
    )

    print(
        "Classification evaluation completed | "
        f"Evaluated terms: {metrics['evaluated_terms']} | "
        f"Correct: {metrics['correct_terms']} | "
        f"Incorrect: {metrics['incorrect_terms']} | "
        f"Accuracy: {metrics['accuracy']:.4f}"
    )

    print(
        report.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"Results saved to: {output_directory}")


if __name__ == "__main__":
    main()