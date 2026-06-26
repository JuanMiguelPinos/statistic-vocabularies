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
    """Guarda el progreso de la anotación manual."""
    sample.to_csv(
        sample_path,
        index=False,
        encoding="utf-8-sig",
    )


def update_correctness(
    sample: pd.DataFrame,
    row_index: int,
) -> None:
    """Indica si la predicción coincide con la categoría manual."""
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
    """Muestra las categorías disponibles."""
    print("=" * 80)
    print("ANOTACIÓN MANUAL DE LA MUESTRA")
    print("=" * 80)
    print()
    print("Categorías:")
    print("  1 = measure")
    print("      Qué se mide, cuenta, calcula o estima.")
    print()
    print("  2 = dimension_name")
    print("      Nombre de un criterio o variable de clasificación.")
    print()
    print("  3 = dimension_value")
    print("      Valor concreto de una dimensión o categoría.")
    print()
    print("  4 = unit")
    print("      Forma numérica en la que se expresa una medida.")
    print()
    print("  5 = other")
    print("      Texto vacío, corrupto o imposible de clasificar.")
    print()
    print("Controles:")
    print("  Intro = aceptar la categoría predicha")
    print("  p     = volver a la fila anterior")
    print("  q     = guardar y salir")
    print()


def main() -> None:
    sample_path = (
        PROJECT_ROOT
        / "evaluation"
        / "gold_sample.csv"
    )

    if not sample_path.exists():
        raise FileNotFoundError(
            "No se encuentra evaluation/gold_sample.csv. "
            "Ejecuta primero python preparar_evaluacion.py."
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
            "Faltan columnas en gold_sample.csv: "
            + ", ".join(sorted(missing_columns))
        )

    print_instructions()

    annotated_mask = (
        sample["gold_category"]
        .astype(str)
        .str.strip()
        .ne("")
    )

    annotated_count = int(annotated_mask.sum())

    print(
        f"Progreso existente: {annotated_count}/{len(sample)}"
    )

    if annotated_count == len(sample):
        print("La muestra ya está completamente anotada.")
        return

    unannotated_indices = sample.index[
        ~annotated_mask
    ].tolist()

    current_position = unannotated_indices[0]

    while 0 <= current_position < len(sample):
        row = sample.loc[current_position]

        print()
        print("=" * 80)
        print(
            f"MUESTRA {row['sample_id']} DE {len(sample)}"
        )
        print("=" * 80)

        term = str(row["term"]).strip()

        if not term:
            term = "[TÉRMINO VACÍO]"

        print(f"Término: {term}")
        print(
            "Predicción: "
            f"{row['predicted_category']}"
        )
        print(f"Confianza: {row['confidence']}")
        print(f"Motivo: {row['reason']}")
        print(f"Columnas: {row['columns']}")

        existing_gold = str(
            row["gold_category"]
        ).strip()

        if existing_gold:
            print(
                f"Categoría manual actual: {existing_gold}"
            )

        print()
        answer = input(
            "Categoría [1-5, Intro, p, q]: "
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

            print()
            print(
                f"Progreso guardado: "
                f"{completed}/{len(sample)}"
            )
            return

        if answer == "p":
            if current_position > 0:
                current_position -= 1
            else:
                print("Ya estás en la primera fila.")

            continue

        if answer == "":
            selected_category = str(
                row["predicted_category"]
            ).strip()

        elif answer in CATEGORIES:
            selected_category = CATEGORIES[answer]

        else:
            print(
                "Opción no válida. Introduce 1, 2, 3, "
                "4, 5, p, q o pulsa Intro."
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

    print()
    print("=" * 80)
    print("ANOTACIÓN FINALIZADA")
    print("=" * 80)
    print(f"Filas anotadas: {completed}/{len(sample)}")
    print(f"Archivo guardado en: {sample_path}")


if __name__ == "__main__":
    main()