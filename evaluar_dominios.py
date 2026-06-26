import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.cluster_measures import DOMAIN_PROFILES, OTHER_DOMAIN
from src.config import PROJECT_ROOT


DOMAINS = list(DOMAIN_PROFILES.keys()) + [OTHER_DOMAIN]


def main() -> None:
    sample_path = (
        PROJECT_ROOT
        / "evaluation"
        / "domain_gold_sample.csv"
    )

    output_directory = (
        PROJECT_ROOT
        / "evaluation"
    )

    figure_directory = (
        output_directory
        / "figures"
    )

    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample = pd.read_csv(
        sample_path,
        keep_default_na=False,
    )

    missing = sample[
        sample["gold_domain"]
        .astype(str)
        .str.strip()
        .eq("")
    ]

    if not missing.empty:
        raise ValueError(
            f"Quedan {len(missing)} términos sin anotar."
        )

    invalid = sample[
        ~sample["gold_domain"].isin(DOMAINS)
    ]

    if not invalid.empty:
        raise ValueError(
            "Hay dominios manuales no válidos: "
            + str(
                sorted(
                    invalid["gold_domain"]
                    .unique()
                    .tolist()
                )
            )
        )

    y_true = sample["gold_domain"]
    y_pred = sample["predicted_domain"]

    sample["is_correct"] = (
        y_true == y_pred
    ).map(
        {
            True: "yes",
            False: "no",
        }
    )

    sample.to_csv(
        sample_path,
        index=False,
        encoding="utf-8-sig",
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=DOMAINS,
        average="macro",
        zero_division=0,
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=DOMAINS,
        average="weighted",
        zero_division=0,
    )

    overall_metrics = pd.DataFrame(
        [
            {
                "evaluated_terms": len(sample),
                "correct_terms": int(
                    (y_true == y_pred).sum()
                ),
                "incorrect_terms": int(
                    (y_true != y_pred).sum()
                ),
                "accuracy": accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "weighted_precision": weighted_precision,
                "weighted_recall": weighted_recall,
                "weighted_f1": weighted_f1,
            }
        ]
    )

    report_dictionary = classification_report(
        y_true,
        y_pred,
        labels=DOMAINS,
        output_dict=True,
        zero_division=0,
    )

    report_rows = []

    for domain in DOMAINS:
        values = report_dictionary[domain]

        report_rows.append(
            {
                "domain": domain,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1_score": values["f1-score"],
                "support": int(values["support"]),
            }
        )

    report = pd.DataFrame(report_rows)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=DOMAINS,
    )

    confusion = pd.DataFrame(
        matrix,
        index=[
            f"true_{domain}"
            for domain in DOMAINS
        ],
        columns=[
            f"predicted_{domain}"
            for domain in DOMAINS
        ],
    )

    overall_metrics.to_csv(
        output_directory
        / "domain_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    report.to_csv(
        output_directory
        / "domain_classification_report.csv",
        index=False,
        encoding="utf-8",
    )

    confusion.to_csv(
        output_directory
        / "domain_confusion_matrix.csv",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(
        figsize=(18, 15)
    )

    image = axis.imshow(matrix)

    axis.set_xticks(
        range(len(DOMAINS))
    )

    axis.set_yticks(
        range(len(DOMAINS))
    )

    axis.set_xticklabels(
        DOMAINS,
        rotation=90,
    )

    axis.set_yticklabels(
        DOMAINS,
    )

    axis.set_xlabel(
        "Predicted domain"
    )

    axis.set_ylabel(
        "Gold domain"
    )

    axis.set_title(
        "Domain classification confusion matrix"
    )

    for row in range(len(DOMAINS)):
        for column in range(len(DOMAINS)):
            axis.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
                fontsize=7,
            )

    figure.colorbar(
        image,
        ax=axis,
    )

    figure.tight_layout()

    figure.savefig(
        figure_directory
        / "domain_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("=" * 80)
    print("EVALUACIÓN DEL PASO 7")
    print("=" * 80)

    print()
    print(f"Términos evaluados: {len(sample)}")
    print(
        "Clasificaciones correctas: "
        f"{int((y_true == y_pred).sum())}"
    )
    print(
        "Clasificaciones incorrectas: "
        f"{int((y_true != y_pred).sum())}"
    )
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy porcentual: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print()
    print("Resultados por dominio:")
    print(
        report.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print("Archivos generados:")
    print("  evaluation/domain_metrics.csv")
    print("  evaluation/domain_classification_report.csv")
    print("  evaluation/domain_confusion_matrix.csv")
    print("  evaluation/figures/domain_confusion_matrix.png")


if __name__ == "__main__":
    main()