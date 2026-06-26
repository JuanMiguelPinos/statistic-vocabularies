from src.config import get_project_paths, load_config
from src.extract_vocabulary import extract_table_strings_chunked
from src.io_utils import list_csv_files


def main() -> None:
    config = load_config()
    paths = get_project_paths(config)

    table_files = list_csv_files(
        paths.tables_full,
        recursive=True,
    )

    table_files = [
        path
        for path in table_files
        if path.name.casefold() != "aact_ali01.csv"
    ]

    if not table_files:
        raise FileNotFoundError(
            "No se encontraron tablas completas."
        )

    largest_table = max(
        table_files,
        key=lambda path: path.stat().st_size,
    )

    size_gb = largest_table.stat().st_size / (1024 ** 3)

    print(f"Tabla: {largest_table.name}")
    print(f"Tamaño: {size_gb:.2f} GB")
    print("Procesando por bloques...")

    (
        records,
        semantic_column_count,
        chunks_processed,
    ) = extract_table_strings_chunked(
        table_path=largest_table,
        chunk_size=100_000,
    )

    print("Prueba terminada correctamente.")
    print(
        f"Columnas descriptivas: "
        f"{semantic_column_count}"
    )
    print(
        f"Bloques procesados: "
        f"{chunks_processed}"
    )
    print(
        f"Términos distintos: "
        f"{len(records)}"
    )


if __name__ == "__main__":
    main()