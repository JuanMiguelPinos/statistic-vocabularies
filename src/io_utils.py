from pathlib import Path
from typing import Any, Iterable

import csv
import sys

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
    """
    Amplía el tamaño máximo de campo admitido por el lector CSV.

    En Windows, sys.maxsize puede superar el límite del tipo C long,
    por lo que se reduce progresivamente si produce OverflowError.
    """
    limit = sys.maxsize

    while limit > 0:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


configure_csv_field_size_limit()


class DataReadingError(RuntimeError):
    """Error producido al leer uno de los recursos del proyecto."""


def list_files(
    directory: Path,
    extensions: Iterable[str],
    recursive: bool = True,
) -> list[Path]:
    """
    Devuelve los archivos de una carpeta con las extensiones indicadas.
    """
    normalized_extensions = {
        extension.lower()
        if extension.startswith(".")
        else f".{extension.lower()}"
        for extension in extensions
    }

    iterator = directory.rglob("*") if recursive else directory.glob("*")

    files = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in normalized_extensions
    ]

    return sorted(files)


def list_csv_files(
    directory: Path,
    recursive: bool = True,
) -> list[Path]:
    """Devuelve todos los CSV encontrados en una carpeta."""
    return list_files(
        directory=directory,
        extensions={".csv"},
        recursive=recursive,
    )


def find_single_file(
    directory: Path,
    extensions: Iterable[str],
) -> Path:
    """
    Encuentra un único archivo dentro de una carpeta.

    Se utiliza para localizar el CSV de títulos y el Excel de NUTS.
    """
    files = list_files(
        directory=directory,
        extensions=extensions,
        recursive=True,
    )

    if not files:
        extensions_text = ", ".join(extensions)
        raise FileNotFoundError(
            f"No se encontraron archivos {extensions_text} en {directory}"
        )

    if len(files) > 1:
        names = "\n".join(f"  - {path.name}" for path in files)

        raise DataReadingError(
            f"Se esperaba un único archivo en {directory}, "
            f"pero se encontraron {len(files)}:\n{names}"
        )

    return files[0]

def detect_compression(path: Path) -> str | None:
    """
    Detecta si un archivo está comprimido como GZIP aunque termine en .csv.

    Los archivos GZIP empiezan por los bytes 1F 8B.
    """
    try:
        with path.open("rb") as file:
            magic_bytes = file.read(2)

        if magic_bytes == b"\x1f\x8b":
            return "gzip"

        return None

    except OSError as exc:
        raise DataReadingError(
            f"No se pudo inspeccionar el archivo: {path}"
        ) from exc


def read_csv_flexible(
    path: Path,
    **read_kwargs: Any,
) -> Any:
    """
    Lee un CSV probando distintas codificaciones y separadores.

    Admite todos los argumentos habituales de pandas.read_csv,
    incluyendo nrows, usecols, chunksize, dtype y keep_default_na.

    Cuando se solicita chunksize, devuelve un TextFileReader que
    permite procesar el archivo por bloques.
    """
    errors: list[str] = []
    compression = detect_compression(path)

    candidates: list[
        tuple[int, str, str | None, str]
    ] = []

    # Primero se hace una lectura pequeña para identificar
    # la combinación correcta de codificación y separador.
    for encoding in CSV_ENCODINGS:
        for separator in CSV_SEPARATORS:
            engine = (
                "python"
                if separator is None
                else "c"
            )

            probe_options = dict(read_kwargs)

            # El sondeo no debe devolver bloques.
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

            # Siempre se leen unas pocas filas para detectar
            # correctamente las columnas.
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
                    "automático"
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
            f"No se pudo detectar el formato del CSV:\n"
            f"{path}\n"
            f"Últimos intentos:\n{attempts}"
        )

    # Se prueban primero las combinaciones que detectaron
    # un mayor número de columnas.
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

        # El motor C es más rápido, pero puede fallar con líneas
        # extremadamente largas. Para lectura por bloques se usa
        # siempre el motor Python.
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
                "automático"
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
        f"No se pudo leer el CSV:\n{path}\n"
        f"Últimos intentos:\n{attempts}"
    )


def read_titles_file(path: Path) -> pd.DataFrame:
    """Lee el archivo que relaciona tablas y títulos."""
    dataframe = read_csv_flexible(path)

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe


def get_excel_sheet_names(path: Path) -> list[str]:
    """Devuelve los nombres de las hojas de un archivo Excel."""
    try:
        excel_file = pd.ExcelFile(path)
        return excel_file.sheet_names
    except Exception as exc:
        raise DataReadingError(
            f"No se pudieron leer las hojas del Excel NUTS: {path}"
        ) from exc


def read_excel_sheet(
    path: Path,
    sheet_name: str | int = 0,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Lee una hoja de un archivo Excel."""
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
            f"No se pudo leer la hoja {sheet_name!r} de {path}"
        ) from exc


def print_dataframe_preview(
    dataframe: pd.DataFrame,
    rows: int = 5,
) -> None:
    """Imprime una vista previa legible de un DataFrame."""
    if dataframe.empty:
        print("[DataFrame vacío]")
        return

    with pd.option_context(
        "display.max_columns",
        15,
        "display.width",
        180,
        "display.max_colwidth",
        50,
    ):
        print(dataframe.head(rows).to_string(index=False))


def inspect_input_data(
    paths: ProjectPaths,
    number_of_table_samples: int = 3,
) -> None:
    """
    Inspecciona los tres recursos principales antes de implementar
    la extracción definitiva.
    """
    print("=" * 80)
    print("1. TABLAS EUROSTAT")
    print("=" * 80)

    table_files = list_csv_files(paths.tables_small)

    print(f"Carpeta: {paths.tables_small}")
    print(f"Número de tablas encontradas: {len(table_files)}")

    if not table_files:
        raise FileNotFoundError(
            f"No se encontraron tablas CSV en {paths.tables_small}"
        )

    for index, table_path in enumerate(
        table_files[:number_of_table_samples],
        start=1,
    ):
        print()
        print(f"Tabla de muestra {index}: {table_path.name}")

        table = read_csv_flexible(
            table_path,
            nrows=5,
        )

        print(f"Dimensiones de la muestra: {table.shape}")
        print(f"Columnas detectadas: {list(table.columns)}")
        print_dataframe_preview(table)

    print()
    print("=" * 80)
    print("2. ARCHIVO DE TÍTULOS")
    print("=" * 80)

    titles_path = find_single_file(
        paths.titles,
        extensions={".csv"},
    )

    print(f"Archivo: {titles_path}")

    titles = read_titles_file(titles_path)

    print(f"Dimensiones: {titles.shape}")
    print(f"Columnas: {list(titles.columns)}")
    print_dataframe_preview(titles)

    print()
    print("=" * 80)
    print("3. DICCIONARIO NUTS")
    print("=" * 80)

    nuts_path = find_single_file(
        paths.nuts,
        extensions={".xlsx", ".xls"},
    )

    print(f"Archivo: {nuts_path}")

    sheet_names = get_excel_sheet_names(nuts_path)

    print(f"Hojas encontradas: {sheet_names}")

    for sheet_name in sheet_names:
        print()
        print(f"Hoja: {sheet_name}")

        preview = read_excel_sheet(
            nuts_path,
            sheet_name=sheet_name,
            nrows=5,
        )

        print(f"Columnas: {list(preview.columns)}")
        print_dataframe_preview(preview)

    print()
    print("=" * 80)
    print("INSPECCIÓN FINALIZADA CORRECTAMENTE")
    print("=" * 80)