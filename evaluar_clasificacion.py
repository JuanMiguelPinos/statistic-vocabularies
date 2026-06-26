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
    """Guarda una representación gráfica de la matriz de confusión."""
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
    """Actualiza la columna is_correct a partir de la anotación."""
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
            "No se encuentra evaluation/gold_sample.csv."
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
            "Todavía existen términos sin anotar: "
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
            "Se encontraron categorías manuales no válidas: "
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

    print("=" * 80)
    print("EVALUACIÓN DEL PASO 6")
    print("=" * 80)

    print()
    print("Métricas generales:")
    print(
        f"  Términos evaluados: "
        f"{metrics['evaluated_terms']}"
    )
    print(
        f"  Clasificaciones correctas: "
        f"{metrics['correct_terms']}"
    )
    print(
        f"  Clasificaciones incorrectas: "
        f"{metrics['incorrect_terms']}"
    )
    print(
        f"  Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )
    print(
        f"  Accuracy porcentual: "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print()
    print("Resultados por categoría:")
    print(
        report.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print("Matriz de confusión:")
    print(
        confusion.to_string()
    )

    print()
    print("Archivos generados:")

    generated_files = [
        output_directory
        / "classification_metrics.csv",
        output_directory
        / "classification_report.csv",
        output_directory
        / "classification_confusion_matrix.csv",
        figure_path,
    ]

    for path in generated_files:
        print(f"  - {path}")


if __name__ == "__main__":
    main()