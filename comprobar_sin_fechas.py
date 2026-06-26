import pandas as pd

from src.config import get_project_paths, load_config
from src.io_utils import list_csv_files, read_csv_flexible


def main() -> None:
    config = load_config()
    paths = get_project_paths(config)

    summary_path = (
        paths.intermediate / "table_dates_summary.csv"
    )

    summary = pd.read_csv(summary_path)

    without_dates = summary[
        (summary["status"] == "ok")
        & (summary["time_interval_count"] == 0)
    ]

    print(f"Tablas sin fechas: {len(without_dates)}")
    print("=" * 80)

    table_files = {
        path.name: path
        for path in list_csv_files(
            paths.tables_small,
            recursive=True,
        )
    }

    diagnostic_records = []

    for filename in without_dates["filename"]:
        path = table_files.get(filename)

        if path is None:
            continue

        dataframe = read_csv_flexible(
            path,
            nrows=2,
        )

        columns = [
            str(column)
            for column in dataframe.columns
        ]

        diagnostic_records.append(
            {
                "filename": filename,
                "columns": " | ".join(columns),
            }
        )

    diagnostic = pd.DataFrame(diagnostic_records)

    output_path = (
        paths.intermediate
        / "tables_without_dates_headers.csv"
    )

    diagnostic.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print("\nPrimeras 15 tablas:\n")

    for record in diagnostic_records[:15]:
        print(f"ARCHIVO: {record['filename']}")
        print(f"COLUMNAS: {record['columns']}")
        print("-" * 80)

    print(f"\nDiagnóstico guardado en:\n{output_path}")


if __name__ == "__main__":
    main()