from src.classify_terms import (
    classify_global_vocabulary,
    save_classification_results,
)
from src.cluster_measures import (
    cluster_measures_by_domain,
    save_domain_results,
)
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


def run_pipeline() -> None:
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
            "dataset_mode must be 'small' or 'full'."
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
            "Excluded files: "
            + ", ".join(sorted(excluded_files))
        )

    sample_limit = processing_config.get(
        "sample_limit"
    )

    if sample_limit is not None:
        sample_limit = int(sample_limit)

        if sample_limit <= 0:
            raise ValueError(
                "processing.sample_limit must be "
                "greater than zero or null."
            )

        table_files = sorted(
            table_files,
            key=lambda path: path.stat().st_size,
        )[:sample_limit]

        print(
            f"Execution limited to the {sample_limit} "
            "smallest tables."
        )

    if not table_files:
        raise FileNotFoundError(
            f"No tables were found in {tables_directory}"
        )

    print(
        f"Dataset mode: {dataset_mode} | "
        f"Tables: {len(table_files)} | "
        f"Chunk size: {chunk_size:,}"
    )

    intervals, time_summary = build_table_time_vocabulary(
        table_files
    )

    save_time_results(
        intervals=intervals,
        summary=time_summary,
        output_directory=paths.intermediate,
    )

    print(
        "Step 1 completed | "
        f"Successful tables: "
        f"{int((time_summary['status'] == 'ok').sum())} | "
        f"Errors: "
        f"{int((time_summary['status'] == 'error').sum())} | "
        f"Tables without dates: "
        f"{int((time_summary['time_interval_count'] == 0).sum())} | "
        f"Intervals: {len(intervals)}"
    )

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

    print(
        "Step 2 completed | "
        f"Successful tables: "
        f"{int((vocabulary_summary['status'] == 'ok').sum())} | "
        f"Errors: "
        f"{int((vocabulary_summary['status'] == 'error').sum())} | "
        f"Table-term pairs: {len(table_vocabulary)} | "
        f"Global terms: {len(global_vocabulary)}"
    )

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

    print(
        "Step 3 completed | "
        f"Dictionary entries: {len(geography_dictionary)} | "
        f"Geographical table-term pairs: "
        f"{len(geographical_terms)} | "
        f"Distinct geographies: "
        f"{geographical_terms['normalized_term'].nunique()} | "
        f"Remaining table-term pairs: "
        f"{len(vocabulary_without_geo)} | "
        f"Remaining global terms: "
        f"{len(global_vocabulary_without_geo)}"
    )

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

    print(
        "Steps 4 and 5 completed | "
        f"Matched titles: {matched_titles} | "
        f"Missing titles: {missing_titles} | "
        f"Title dates: {len(title_dates)} | "
        f"Title geographies: {len(title_geographies)} | "
        f"Residual titles: {len(title_vocabulary)} | "
        f"Final table-term pairs: "
        f"{len(final_table_vocabulary)} | "
        f"Final global terms: "
        f"{len(final_global_vocabulary)}"
    )

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

    print(
        "Step 6 completed | "
        f"Measures: {category_counts.get('measure', 0)} | "
        f"Dimension names: "
        f"{category_counts.get('dimension_name', 0)} | "
        f"Dimension values: "
        f"{category_counts.get('dimension_value', 0)} | "
        f"Units: {category_counts.get('unit', 0)} | "
        f"Other: {category_counts.get('other', 0)} | "
        f"Total: {len(classification)}"
    )

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

    print(
        "Step 7 completed | "
        f"Measures assigned: {len(domain_assignments)} | "
        f"Domains: "
        f"{domain_assignments['domain'].nunique()} | "
        f"High confidence: "
        f"{int((domain_assignments['confidence'] == 'high').sum())} | "
        f"Medium confidence: "
        f"{int((domain_assignments['confidence'] == 'medium').sum())} | "
        f"Low confidence: "
        f"{int((domain_assignments['confidence'] == 'low').sum())}"
    )

    print("Pipeline completed successfully.")