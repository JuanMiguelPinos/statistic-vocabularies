import pandas as pd

from src.config import PROJECT_ROOT


DOMAINS = {
    "1": "Agriculture, forestry and fisheries",
    "2": "Business, industry and trade",
    "3": "Crime, justice and safety",
    "4": "Demography and population",
    "5": "Economy and national accounts",
    "6": "Education and training",
    "7": "Energy",
    "8": "Environment and climate",
    "9": "Government and public finance",
    "10": "Health",
    "11": "Housing and construction",
    "12": "Income, poverty and living conditions",
    "13": "Labour market",
    "14": "Other or multidisciplinary",
    "15": "Science, technology and digital society",
    "16": "Tourism",
    "17": "Transport and mobility",
}


def print_domain_menu() -> None:
    """Muestra los dominios disponibles."""
    print()
    print("Dominios:")

    for number, domain in DOMAINS.items():
        print(f"  {number:>2} = {domain}")

    print()
    print("Controles:")
    print("  Intro = aceptar el dominio predicho")
    print("  ?     = volver a mostrar los dominios")
    print("  p     = volver a la fila anterior")
    print("  q     = guardar y salir")


def save_progress(
    sample: pd.DataFrame,
    sample_path,
) -> None:
    """Guarda el progreso de la anotación."""
    sample.to_csv(
        sample_path,
        index=False,
        encoding="utf-8-sig",
    )


def update_correctness(
    sample: pd.DataFrame,
    row_index: int,
) -> None:
    """Actualiza la columna is_correct."""
    predicted = str(
        sample.at[row_index, "predicted_domain"]
    ).strip()

    gold = str(
        sample.at[row_index, "gold_domain"]
    ).strip()

    sample.at[row_index, "is_correct"] = (
        "yes"
        if predicted == gold
        else "no"
    )


def main() -> None:
    sample_path = (
        PROJECT_ROOT
        / "evaluation"
        / "domain_gold_sample.csv"
    )

    if not sample_path.exists():
        raise FileNotFoundError(
            "No se encuentra evaluation/domain_gold_sample.csv. "
            "Ejecuta primero python preparar_evaluacion_dominios.py."
        )

    sample = pd.read_csv(
        sample_path,
        keep_default_na=False,
    )

    required_columns = {
        "sample_id",
        "term",
        "predicted_domain",
        "domain_score",
        "second_domain",
        "confidence",
        "reason",
        "gold_domain",
        "is_correct",
        "notes",
    }

    missing_columns = required_columns.difference(
        sample.columns
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas en domain_gold_sample.csv: "
            + ", ".join(sorted(missing_columns))
        )

    print("=" * 80)
    print("ANOTACIÓN MANUAL DE DOMINIOS")
    print("=" * 80)

    print_domain_menu()

    annotated_mask = (
        sample["gold_domain"]
        .astype(str)
        .str.strip()
        .ne("")
    )

    annotated_count = int(
        annotated_mask.sum()
    )

    print(
        f"\nProgreso existente: "
        f"{annotated_count}/{len(sample)}"
    )

    if annotated_count == len(sample):
        print(
            "La muestra ya está completamente anotada."
        )
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
            f"MUESTRA {row['sample_id']} "
            f"DE {len(sample)}"
        )
        print("=" * 80)

        print(f"Término: {row['term']}")
        print(
            f"Dominio predicho: "
            f"{row['predicted_domain']}"
        )
        print(
            f"Segundo dominio: "
            f"{row['second_domain']}"
        )
        print(
            f"Confianza: {row['confidence']}"
        )
        print(
            f"Puntuación: {row['domain_score']}"
        )
        print(f"Motivo: {row['reason']}")

        existing_gold = str(
            row["gold_domain"]
        ).strip()

        if existing_gold:
            print(
                f"Dominio manual actual: "
                f"{existing_gold}"
            )

        answer = input(
            "\nDominio [1-17, Intro, ?, p, q]: "
        ).strip().casefold()

        if answer == "q":
            save_progress(
                sample,
                sample_path,
            )

            completed = int(
                sample["gold_domain"]
                .astype(str)
                .str.strip()
                .ne("")
                .sum()
            )

            print(
                f"\nProgreso guardado: "
                f"{completed}/{len(sample)}"
            )
            return

        if answer == "?":
            print_domain_menu()
            continue

        if answer == "p":
            if current_position > 0:
                current_position -= 1
            else:
                print(
                    "Ya estás en la primera fila."
                )

            continue

        if answer == "":
            selected_domain = str(
                row["predicted_domain"]
            ).strip()

        elif answer in DOMAINS:
            selected_domain = DOMAINS[answer]

        else:
            print(
                "Opción no válida. Introduce un número "
                "entre 1 y 17, Intro, ?, p o q."
            )
            continue

        sample.at[
            current_position,
            "gold_domain",
        ] = selected_domain

        update_correctness(
            sample=sample,
            row_index=current_position,
        )

        save_progress(
            sample,
            sample_path,
        )

        current_position += 1

    completed = int(
        sample["gold_domain"]
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    print()
    print("=" * 80)
    print("ANOTACIÓN DE DOMINIOS FINALIZADA")
    print("=" * 80)
    print(
        f"Filas anotadas: "
        f"{completed}/{len(sample)}"
    )
    print(
        f"Archivo guardado en: "
        f"{sample_path}"
    )


if __name__ == "__main__":
    main()