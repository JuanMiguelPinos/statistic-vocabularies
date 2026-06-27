import csv
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.config import ProjectPaths


CSV_ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "latin-1",
)

CSV_SEPARATORS = (
    "\t",
    ";",
    ",",
    None,
)


def configure_csv_field_size_limit() -> None:
    limit = sys.maxsize

    while limit > 0:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


configure_csv_field_size_limit()


class DataReadingError(RuntimeError):
    pass


def list_files(
    directory: Path,
    extensions: Iterable[str],
    recursive: bool = True,
) -> list[Path]:
    normalized_extensions = {
        extension.lower()
        if extension.startswith(".")
        else f".{extension.lower()}"
        for extension in extensions
    }

    iterator = (
        directory.rglob("*")
        if recursive
        else directory.glob("*")
    )

    files = [
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in normalized_extensions
    ]

    return sorted(files)


def list_csv_files(
    directory: Path,
    recursive: bool = True,
) -> list[Path]:
    return list_files(
        directory=directory,
        extensions={".csv"},
        recursive=recursive,
    )


def find_single_file(
    directory: Path,
    extensions: Iterable[str],
) -> Path:
    files = list_files(
        directory=directory,
        extensions=extensions,
        recursive=True,
    )

    if not files:
        extensions_text = ", ".join(extensions)

        raise FileNotFoundError(
            f"No {extensions_text} files were found in {directory}"
        )

    if len(files) > 1:
        names = "\n".join(
            f"  - {path.name}"
            for path in files
        )

        raise DataReadingError(
            f"Expected one file in {directory}, "
            f"but found {len(files)}:\n{names}"
        )

    return files[0]


def detect_compression(path: Path) -> str | None:
    try:
        with path.open("rb") as file:
            magic_bytes = file.read(2)

        if magic_bytes == b"\x1f\x8b":
            return "gzip"

        return None

    except OSError as exc:
        raise DataReadingError(
            f"Could not inspect file: {path}"
        ) from exc


def read_csv_flexible(
    path: Path,
    **read_kwargs: Any,
) -> Any:
    errors: list[str] = []
    compression = detect_compression(path)

    candidates: list[
        tuple[int, str, str | None, str]
    ] = []

    for encoding in CSV_ENCODINGS:
        for separator in CSV_SEPARATORS:
            engine = (
                "python"
                if separator is None
                else "c"
            )

            probe_options = dict(read_kwargs)

            probe_options.pop(
                "chunksize",
                None,
            )

            probe_options.pop(
                "iterator",
                None,
            )

            probe_options.pop(
                "usecols",
                None,
            )

            probe_options["nrows"] = 5

            probe_options.setdefault(
                "dtype",
                str,
            )

            probe_options.setdefault(
                "keep_default_na",
                False,
            )

            probe_options.setdefault(
                "on_bad_lines",
                "skip",
            )

            probe_options.update(
                {
                    "sep": separator,
                    "encoding": encoding,
                    "compression": compression,
                    "engine": engine,
                }
            )

            try:
                probe = pd.read_csv(
                    path,
                    **probe_options,
                )

                if probe.shape[1] > 0:
                    candidates.append(
                        (
                            int(probe.shape[1]),
                            encoding,
                            separator,
                            engine,
                        )
                    )

            except Exception as exc:
                separator_name = (
                    "automatic"
                    if separator is None
                    else repr(separator)
                )

                errors.append(
                    f"encoding={encoding}, "
                    f"separator={separator_name}, "
                    f"compression={compression}: {exc}"
                )

    if not candidates:
        attempts = "\n".join(
            errors[-5:]
        )

        raise DataReadingError(
            f"Could not detect CSV format:\n"
            f"{path}\n"
            f"Last attempts:\n{attempts}"
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    chunked_read = (
        read_kwargs.get("chunksize") is not None
    )

    for (
        _,
        encoding,
        separator,
        detected_engine,
    ) in candidates:
        options = dict(read_kwargs)

        options.setdefault(
            "dtype",
            str,
        )

        options.setdefault(
            "keep_default_na",
            False,
        )

        options.setdefault(
            "on_bad_lines",
            "skip",
        )

        selected_engine = (
            "python"
            if chunked_read
            else detected_engine
        )

        options.update(
            {
                "sep": separator,
                "encoding": encoding,
                "compression": compression,
                "engine": selected_engine,
            }
        )

        try:
            result = pd.read_csv(
                path,
                **options,
            )

            if chunked_read:
                return result

            if result.shape[1] > 0:
                return result

        except Exception as exc:
            separator_name = (
                "automatic"
                if separator is None
                else repr(separator)
            )

            errors.append(
                f"encoding={encoding}, "
                f"separator={separator_name}, "
                f"compression={compression}, "
                f"engine={selected_engine}: {exc}"
            )

    attempts = "\n".join(
        errors[-5:]
    )

    raise DataReadingError(
        f"Could not read CSV file:\n"
        f"{path}\n"
        f"Last attempts:\n{attempts}"
    )


def read_titles_file(path: Path) -> pd.DataFrame:
    dataframe = read_csv_flexible(path)

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe


def get_excel_sheet_names(path: Path) -> list[str]:
    try:
        excel_file = pd.ExcelFile(path)
        return excel_file.sheet_names

    except Exception as exc:
        raise DataReadingError(
            f"Could not read Excel sheet names: {path}"
        ) from exc


def read_excel_sheet(
    path: Path,
    sheet_name: str | int = 0,
    nrows: int | None = None,
) -> pd.DataFrame:
    try:
        return pd.read_excel(
            path,
            sheet_name=sheet_name,
            nrows=nrows,
            dtype=str,
            keep_default_na=False,
        )

    except Exception as exc:
        raise DataReadingError(
            f"Could not read sheet {sheet_name!r} from {path}"
        ) from exc


def print_dataframe_preview(
    dataframe: pd.DataFrame,
    rows: int = 5,
) -> None:
    if dataframe.empty:
        print("[Empty DataFrame]")
        return

    with pd.option_context(
        "display.max_columns",
        15,
        "display.width",
        180,
        "display.max_colwidth",
        50,
    ):
        print(
            dataframe.head(rows).to_string(
                index=False
            )
        )


def inspect_input_data(
    paths: ProjectPaths,
    number_of_table_samples: int = 3,
) -> None:
    table_files = list_csv_files(
        paths.tables_small
    )

    if not table_files:
        raise FileNotFoundError(
            f"No CSV tables were found in {paths.tables_small}"
        )

    print(
        f"Eurostat tables: {len(table_files)} "
        f"files in {paths.tables_small}"
    )

    for table_path in table_files[
        :number_of_table_samples
    ]:
        table = read_csv_flexible(
            table_path,
            nrows=5,
        )

        print(
            f"\nTable: {table_path.name} "
            f"shape={table.shape}"
        )

        print_dataframe_preview(table)

    titles_path = find_single_file(
        paths.titles,
        extensions={".csv"},
    )

    titles = read_titles_file(
        titles_path
    )

    print(
        f"\nTitles file: {titles_path} "
        f"shape={titles.shape}"
    )

    print_dataframe_preview(titles)

    nuts_path = find_single_file(
        paths.nuts,
        extensions={".xlsx", ".xls"},
    )

    sheet_names = get_excel_sheet_names(
        nuts_path
    )

    print(
        f"\nNUTS file: {nuts_path} "
        f"sheets={len(sheet_names)}"
    )

    for sheet_name in sheet_names:
        preview = read_excel_sheet(
            nuts_path,
            sheet_name=sheet_name,
            nrows=5,
        )

        print(
            f"\nSheet: {sheet_name} "
            f"shape={preview.shape}"
        )

        print_dataframe_preview(preview)