from src.config import (
    ensure_output_directories,
    get_project_paths,
    load_config,
)
from src.extract_geo import (
    build_geography_dictionary,
    build_geography_summary,
    identify_geographical_terms,
    remove_geographies_from_vocabulary,
    save_geography_results,
)
from src.extract_time import (
    build_table_time_vocabulary,
    save_time_results,
)
from src.extract_vocabulary import (
    build_global_table_vocabulary,
    build_table_string_vocabulary,
    save_vocabulary_results,
)

from src.io_utils import (
    find_single_file,
    list_csv_files,
    read_titles_file,
)
from src.process_titles import (
    combine_table_and_title_vocabulary,
    process_titles,
    save_title_results,
)

from src.classify_terms import (
    classify_global_vocabulary,
    save_classification_results,
)

from src.cluster_measures import (
    cluster_measures_by_domain,
    save_domain_results,
)

def run_pipeline() -> None:
    """Ejecuta los pasos implementados del proyecto."""
    print("=" * 80)
    print("STAR STATISTICAL VOCABULARIES")
    print("=" * 80)

    config = load_config()
    paths = get_project_paths(config)

    processing_config = config.get(
        "processing",
        {},
    )

    chunk_size = int(
        processing_config.get(
            "chunk_size",
            100_000,
        )
    )

    print(
        f"Tamaño de bloque: {chunk_size:,} filas"
    )

    ensure_output_directories(paths)

    dataset_mode = str(
        config.get("dataset_mode", "small")
    ).strip().casefold()

    if dataset_mode == "small":
        tables_directory = paths.tables_small
    elif dataset_mode == "full":
        tables_directory = paths.tables_full
    else:
        raise ValueError(
            "dataset_mode debe ser 'small' o 'full'."
        )

    table_files = list_csv_files(
        tables_directory,
        recursive=True,
    )

    excluded_files = {
        str(filename).strip().casefold()
        for filename in processing_config.get(
            "exclude_files",
            [],
        )
    }

    if excluded_files:
        table_files = [
            path
            for path in table_files
            if path.name.casefold() not in excluded_files
        ]

        print(
            "Archivos excluidos: "
            + ", ".join(sorted(excluded_files))
        )

    sample_limit = processing_config.get(
        "sample_limit"
    )

    if sample_limit is not None:
        sample_limit = int(sample_limit)

        if sample_limit <= 0:
            raise ValueError(
                "processing.sample_limit debe ser "
                "mayor que cero o null."
            )

        # Para la prueba se seleccionan los archivos
        # más pequeños del conjunto completo.
        table_files = sorted(
            table_files,
            key=lambda path: path.stat().st_size,
        )[:sample_limit]

        print(
            "Ejecución limitada a las "
            f"{sample_limit} tablas más pequeñas."
        )

    print(f"Modo del dataset: {dataset_mode}")
    print(f"Directorio de tablas: {tables_directory}")
    print(f"Tablas encontradas: {len(table_files)}")

    if not table_files:
        raise FileNotFoundError(
            f"No se encontraron tablas en {tables_directory}"
        )

    # ============================================================
    # PASO 1
    # ============================================================

    print()
    print("=" * 80)
    print("PASO 1: EXTRACCIÓN DE INTERVALOS TEMPORALES D(t)")
    print("=" * 80)

    intervals, time_summary = build_table_time_vocabulary(
        table_files
    )

    save_time_results(
        intervals=intervals,
        summary=time_summary,
        output_directory=paths.intermediate,
    )

    print()
    print("Resultado del paso 1:")
    print(
        "  Tablas procesadas correctamente: "
        f"{int((time_summary['status'] == 'ok').sum())}"
    )
    print(
        "  Tablas con error: "
        f"{int((time_summary['status'] == 'error').sum())}"
    )
    print(
        "  Tablas sin fechas detectadas: "
        f"{int((time_summary['time_interval_count'] == 0).sum())}"
    )
    print(f"  Intervalos extraídos: {len(intervals)}")

    # ============================================================
    # PASO 2
    # ============================================================

    print()
    print("=" * 80)
    print("PASO 2: EXTRACCIÓN DEL VOCABULARIO S(t)")
    print("=" * 80)

    table_vocabulary, vocabulary_summary = (
        build_table_string_vocabulary(
            table_files=table_files,
            chunk_size=chunk_size,
        )
    )

    global_vocabulary = build_global_table_vocabulary(
        table_vocabulary
    )

    save_vocabulary_results(
        table_vocabulary=table_vocabulary,
        global_vocabulary=global_vocabulary,
        summary=vocabulary_summary,
        output_directory=paths.intermediate,
    )

    print()
    print("Resultado del paso 2:")
    print(
        "  Tablas procesadas correctamente: "
        f"{int((vocabulary_summary['status'] == 'ok').sum())}"
    )
    print(
        "  Tablas con error: "
        f"{int((vocabulary_summary['status'] == 'error').sum())}"
    )
    print(
        "  Términos distintos tabla-término: "
        f"{len(table_vocabulary)}"
    )
    print(
        "  Términos distintos globales: "
        f"{len(global_vocabulary)}"
    )

    # ============================================================
    # PASO 3
    # ============================================================

    print()
    print("=" * 80)
    print("PASO 3: IDENTIFICACIÓN DE GEOGRAFÍAS Geo(t)")
    print("=" * 80)

    nuts_path = find_single_file(
        paths.nuts,
        extensions={".xlsx", ".xls"},
    )

    geography_dictionary = build_geography_dictionary(
        nuts_path
    )

    geographical_terms = identify_geographical_terms(
        table_vocabulary=table_vocabulary,
        geography_dictionary=geography_dictionary,
    )

    vocabulary_without_geo = (
        remove_geographies_from_vocabulary(
            table_vocabulary=table_vocabulary,
            geographical_terms=geographical_terms,
        )
    )

    global_vocabulary_without_geo = (
        build_global_table_vocabulary(
            vocabulary_without_geo
        )
    )

    geography_summary = build_geography_summary(
        table_vocabulary=table_vocabulary,
        vocabulary_without_geo=vocabulary_without_geo,
        geographical_terms=geographical_terms,
    )

    save_geography_results(
        geography_dictionary=geography_dictionary,
        geographical_terms=geographical_terms,
        vocabulary_without_geo=vocabulary_without_geo,
        global_vocabulary_without_geo=(
            global_vocabulary_without_geo
        ),
        summary=geography_summary,
        output_directory=paths.intermediate,
    )

    print()
    print("Resultado del paso 3:")
    print(
        "  Entradas del diccionario geográfico: "
        f"{len(geography_dictionary)}"
    )
    print(
        "  Geografías tabla-término detectadas: "
        f"{len(geographical_terms)}"
    )
    print(
        "  Geografías globales distintas: "
        f"{geographical_terms['normalized_term'].nunique()}"
    )
    print(
        "  Términos restantes tabla-término: "
        f"{len(vocabulary_without_geo)}"
    )
    print(
        "  Términos restantes globales: "
        f"{len(global_vocabulary_without_geo)}"
    )

    print()
    print("=" * 80)
    print("PASOS 4 Y 5: TÍTULOS Y VOCABULARIO GLOBAL V")
    print("=" * 80)

    titles_path = find_single_file(
        paths.titles,
        extensions={".csv"},
    )

    titles = read_titles_file(titles_path)

    (
        title_processing,
        title_vocabulary,
        title_dates,
        title_geographies,
    ) = process_titles(
        titles=titles,
        table_filenames={
            table_path.name
            for table_path in table_files
        },
        geography_dictionary=geography_dictionary,
    )

    final_table_vocabulary = (
        combine_table_and_title_vocabulary(
            vocabulary_without_geo=(
                vocabulary_without_geo
            ),
            title_vocabulary=title_vocabulary,
        )
    )

    final_global_vocabulary = (
        build_global_table_vocabulary(
            final_table_vocabulary
        )
    )

    save_title_results(
        processing=title_processing,
        title_vocabulary=title_vocabulary,
        title_dates=title_dates,
        title_geographies=title_geographies,
        final_table_vocabulary=(
            final_table_vocabulary
        ),
        final_global_vocabulary=(
            final_global_vocabulary
        ),
        output_directory=paths.intermediate,
    )

    matched_titles = int(
        (
            title_processing["status"]
            == "ok"
        ).sum()
    )

    missing_titles = int(
        (
            title_processing["status"]
            == "missing_title"
        ).sum()
    )

    print()
    print("Resultado de los pasos 4 y 5:")
    print(
        f"  Títulos asociados: {matched_titles}"
    )
    print(
        f"  Tablas sin título: {missing_titles}"
    )
    print(
        "  Fechas extraídas de títulos: "
        f"{len(title_dates)}"
    )
    print(
        "  Geografías extraídas de títulos: "
        f"{len(title_geographies)}"
    )
    print(
        "  Títulos residuales añadidos: "
        f"{len(title_vocabulary)}"
    )
    print(
        "  Pares tabla-término en V(t): "
        f"{len(final_table_vocabulary)}"
    )
    print(
        "  Términos distintos en V: "
        f"{len(final_global_vocabulary)}"
    )

    print()
    print("=" * 80)
    print("PASO 6: CLASIFICACIÓN DEL VOCABULARIO V")
    print("=" * 80)

    classification, classification_summary = (
        classify_global_vocabulary(
            table_vocabulary=final_table_vocabulary,
        )
    )

    save_classification_results(
        classification=classification,
        summary=classification_summary,
        output_directory=paths.outputs,
    )

    category_counts = (
        classification["category"]
        .value_counts()
        .to_dict()
    )

    print()
    print("Resultado del paso 6:")
    print(
        "  Medidas: "
        f"{category_counts.get('measure', 0)}"
    )
    print(
        "  Nombres de dimensiones: "
        f"{category_counts.get('dimension_name', 0)}"
    )
    print(
        "  Valores de dimensiones: "
        f"{category_counts.get('dimension_value', 0)}"
    )
    print(
        "  Unidades: "
        f"{category_counts.get('unit', 0)}"
    )
    print(
        "  Other: "
        f"{category_counts.get('other', 0)}"
    )
    print(
        "  Total clasificado: "
        f"{len(classification)}"
    )

    print()
    print("=" * 80)
    print("PASO 7: AGRUPACIÓN DE MEDIDAS POR DOMINIOS")
    print("=" * 80)

    measures = classification[
        classification["category"] == "measure"
    ].copy()

    (
        domain_assignments,
        domain_summary,
        domain_examples,
    ) = cluster_measures_by_domain(
        measures=measures,
    )

    save_domain_results(
        assignments=domain_assignments,
        summary=domain_summary,
        examples=domain_examples,
        output_directory=paths.outputs,
    )

    print()
    print("Resultado del paso 7:")
    print(
        f"  Medidas agrupadas: "
        f"{len(domain_assignments)}"
    )
    print(
        f"  Dominios utilizados: "
        f"{domain_assignments['domain'].nunique()}"
    )
    print(
        "  Asignaciones con confianza alta: "
        f"{int((domain_assignments['confidence'] == 'high').sum())}"
    )
    print(
        "  Asignaciones con confianza media: "
        f"{int((domain_assignments['confidence'] == 'medium').sum())}"
    )
    print(
        "  Asignaciones con confianza baja: "
        f"{int((domain_assignments['confidence'] == 'low').sum())}"
    )

    print()
    print("Distribución por dominio:")
    print(
        domain_summary[
            [
                "domain",
                "measure_count",
                "average_score",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 80)
    print("PASOS 1 A 7 FINALIZADOS")
    print("=" * 80)