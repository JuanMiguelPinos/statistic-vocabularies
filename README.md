# STAR Statistical Vocabularies

Automatic extraction, classification and organisation of statistical vocabulary from Eurostat tables.

## Repository

https://github.com/JuanMiguelPinos/statistic-vocabularies

## Overview

This project implements a reproducible pipeline for extracting and organising statistical vocabulary from a large collection of Eurostat tables.

The pipeline analyses descriptive table columns and official titles to identify temporal expressions, geographical entities and statistical terminology. It constructs a global vocabulary, classifies each term into a semantic category and assigns the detected measures to statistical domains.

The final full execution processed 7,604 tables successfully and reported no processing errors. One source table was excluded because its internal GZIP stream was incomplete.

The optional semantic-relations stage was not implemented.

## Pipeline

The implementation covers Steps 1–7 of the assignment.

### Step 1 — Time extraction

For each table (t), the pipeline detects temporal expressions and converts them into normalised intervals (D(t)).

The extraction includes individual years, ranges and other supported temporal patterns found in descriptive table content.

### Step 2 — Table vocabulary extraction

The pipeline reads the descriptive columns of each table and constructs a duplicate-free set (S(t)) containing its non-numeric textual values.

Large files are processed in chunks to reduce memory consumption.

### Step 3 — Geography extraction

Geographical terms are detected using the supplied NUTS nomenclature.

For each table, the geographical subset (Geo(t)) is identified and removed from the provisional vocabulary:

```text
V(t) = S(t) - Geo(t)
```

### Steps 4 and 5 — Title processing

Official Eurostat titles are matched to the processed tables.

Dates and geographical expressions found in the titles are removed, while the remaining semantic content is added to the corresponding table vocabulary (V(t)).

The final global vocabulary is obtained by combining the table vocabularies:

```text
V = union of all V(t)
```

### Step 6 — Vocabulary classification

Every term in the global vocabulary is assigned to exactly one of the following categories:

* measure;
* dimension name;
* dimension value;
* unit;
* other.

The classification is based on lexical, structural and contextual heuristics.

### Step 7 — Measure-domain assignment

Every detected measure is assigned to one of seventeen statistical domains.

The assignment combines domain-specific lexical rules with TF-IDF similarity. Each prediction also receives a confidence level:

* high;
* medium;
* low.

### Evaluation

The semantic classification and domain-assignment stages are evaluated separately using manually annotated samples.

The optional Step 8, concerning semantic relationships between terms, was not implemented.

## Final results

The original collection contained 7,605 tables.

The file `aact_ali01.csv` was excluded because its internal GZIP stream ended before the expected end-of-stream marker. The integrity of the downloaded archive had previously been verified, indicating that the issue was associated with the internal source file rather than an incomplete local download.

| Result                                      |     Value |
| ------------------------------------------- | --------: |
| Original tables                             |     7,605 |
| Excluded tables                             |         1 |
| Selected tables                             |     7,604 |
| Successfully processed tables               |     7,604 |
| Tables with processing errors               |         0 |
| Tables without detected dates               |        31 |
| Extracted temporal intervals                |   198,090 |
| Table-term pairs in the initial vocabulary  | 1,151,463 |
| Distinct provisional vocabulary terms       |   159,477 |
| Geographical table-term pairs               |   413,625 |
| Distinct geographical terms                 |     3,558 |
| Remaining non-geographical table-term pairs |   737,838 |
| Remaining non-geographical terms            |   155,919 |
| Official titles available                   |     7,678 |
| Matched table titles                        |     7,604 |
| Missing table titles                        |         0 |
| Dates extracted from titles                 |     1,284 |
| Geographies extracted from titles           |       335 |
| Final table-term pairs                      |   745,418 |
| Distinct terms in the final vocabulary      |   162,236 |

## Vocabulary classification

The 162,236 terms in the final vocabulary were assigned to exactly one semantic category.

| Category         |       Terms | Percentage |
| ---------------- | ----------: | ---------: |
| Measures         |      11,209 |      6.91% |
| Dimension names  |      15,204 |      9.37% |
| Dimension values |     132,992 |     81.97% |
| Units            |       2,830 |      1.74% |
| Other            |           1 |     <0.01% |
| **Total**        | **162,236** |   **100%** |

The category totals satisfy the consistency check:

```text
11,209 + 15,204 + 132,992 + 2,830 + 1 = 162,236
```

## Measure-domain assignment

All 11,209 measures were assigned to one of seventeen statistical domains.

### Confidence distribution

| Confidence | Assignments | Percentage |
| ---------- | ----------: | ---------: |
| High       |       5,851 |     52.20% |
| Medium     |       2,240 |     19.98% |
| Low        |       3,118 |     27.82% |
| **Total**  |  **11,209** |   **100%** |

### Domain distribution

| Domain                                  | Measures | Average score |
| --------------------------------------- | -------: | ------------: |
| Other or multidisciplinary              |    2,844 |        0.0139 |
| Labour market                           |    1,475 |        0.1692 |
| Science, technology and digital society |    1,266 |        0.1377 |
| Education and training                  |    1,042 |        0.2354 |
| Business, industry and trade            |      879 |        0.1953 |
| Demography and population               |      640 |        0.1495 |
| Economy and national accounts           |      557 |        0.1674 |
| Transport and mobility                  |      462 |        0.2545 |
| Income, poverty and living conditions   |      397 |        0.4643 |
| Health                                  |      336 |        0.1488 |
| Agriculture, forestry and fisheries     |      279 |        0.2105 |
| Government and public finance           |      217 |        0.1660 |
| Environment and climate                 |      204 |        0.1488 |
| Energy                                  |      180 |        0.2327 |
| Housing and construction                |      158 |        0.1720 |
| Tourism                                 |      138 |        0.3598 |
| Crime, justice and safety               |      135 |        0.1531 |

## Evaluation

Two manually annotated samples were used to evaluate the semantic stages.

### Vocabulary-classification evaluation

The classification sample contained 81 manually labelled terms.

| Metric                |  Value |
| --------------------- | -----: |
| Evaluated terms       |     81 |
| Correct predictions   |     70 |
| Incorrect predictions |     11 |
| Accuracy              | 86.42% |

| Category        | Precision | Recall | F1-score | Support |
| --------------- | --------: | -----: | -------: | ------: |
| Measure         |    1.0000 | 0.8000 |   0.8889 |      25 |
| Dimension name  |    1.0000 | 1.0000 |   1.0000 |      20 |
| Dimension value |    0.8000 | 0.8000 |   0.8000 |      20 |
| Unit            |    0.6500 | 1.0000 |   0.7879 |      13 |
| Other           |    1.0000 | 0.3333 |   0.5000 |       3 |

Run the evaluation from the project root:

```bash
python evaluate_classification.py
```

The generated files are:

```text
evaluation/classification_metrics.csv
evaluation/classification_report.csv
evaluation/classification_confusion_matrix.csv
evaluation/figures/classification_confusion_matrix.png
```

### Domain-assignment evaluation

The domain sample contained 85 manually labelled measures.

| Metric                |  Value |
| --------------------- | -----: |
| Evaluated measures    |     85 |
| Correct assignments   |     73 |
| Incorrect assignments |     12 |
| Accuracy              | 85.88% |
| Macro F1-score        | 0.8459 |
| Weighted F1-score     | 0.8717 |

Run the evaluation from the project root:

```bash
python evaluate_domains.py
```

The generated files are:

```text
evaluation/domain_metrics.csv
evaluation/domain_classification_report.csv
evaluation/domain_confusion_matrix.csv
evaluation/figures/domain_confusion_matrix.png
```

## Evaluation samples

The manually annotated samples are stored in:

```text
evaluation/gold_sample.csv
evaluation/domain_gold_sample.csv
```

The preparation scripts create new evaluation samples:

```bash
python prepare_classification_evaluation.py
python prepare_domain_evaluation.py
```

These scripts may overwrite the existing sample files. They should not be executed during normal reproduction of the final evaluation unless new samples are intentionally being created.

The annotation interfaces can be launched with:

```bash
python annotate_classification_sample.py
python annotate_domain_sample.py
```

The annotation scripts are only required when a newly prepared sample contains unlabelled rows.

## Repository structure

```text
statistic-vocabularies/
├── config/
│   └── config.yaml
├── evaluation/
│   ├── figures/
│   │   ├── classification_confusion_matrix.png
│   │   └── domain_confusion_matrix.png
│   ├── classification_confusion_matrix.csv
│   ├── classification_metrics.csv
│   ├── classification_report.csv
│   ├── domain_classification_report.csv
│   ├── domain_confusion_matrix.csv
│   ├── domain_gold_sample.csv
│   ├── domain_metrics.csv
│   └── gold_sample.csv
├── report/
│   └── statistic_vocabularies.pdf
├── runs/
│   ├── full/
│   │   ├── logs/
│   │   │   └── full_run_summary.txt
│   │   └── outputs/
│   │       ├── classification_all.csv
│   │       ├── classification_summary.csv
│   │       ├── dimension_names.csv
│   │       ├── dimension_values.csv
│   │       ├── domain_examples.csv
│   │       ├── domain_summary.csv
│   │       ├── measure_domains.csv
│   │       ├── measures.csv
│   │       ├── other.csv
│   │       └── units.csv
│   └── small/
│       └── outputs/
├── src/
│   ├── __init__.py
│   ├── classify_terms.py
│   ├── cluster_measures.py
│   ├── config.py
│   ├── evaluate.py
│   ├── extract_geo.py
│   ├── extract_time.py
│   ├── extract_vocabulary.py
│   ├── io_utils.py
│   ├── pipeline.py
│   └── process_titles.py
├── annotate_classification_sample.py
├── annotate_domain_sample.py
├── evaluate_classification.py
├── evaluate_domains.py
├── main.py
├── prepare_classification_evaluation.py
├── prepare_domain_evaluation.py
├── requirements.txt
├── .gitignore
└── README.md
```

The original datasets, external resources and large intermediate files are intentionally excluded from the repository.

## Requirements

The project was developed and tested with Python 3.11.

Python 3.10 or later is required.

Main dependencies include:

* pandas;
* NumPy;
* scikit-learn;
* openpyxl;
* PyYAML;
* tqdm;
* Unidecode;
* matplotlib.

Create and activate a Conda environment:

```bash
conda create -n star_vocab python=3.11
conda activate star_vocab
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Required data

The original datasets are not stored in the repository because of their size.

Create the following local directory structure:

```text
data/
├── external/
│   └── nuts/
│       └── NUTS2021-NUTS2024.xlsx
└── raw/
    ├── tables_small/
    ├── tables_full/
    └── titles/
        └── file-names_to_titles_eurostat_unfiltered.csv
```

Place the reduced table collection in:

```text
data/raw/tables_small/
```

Place the complete table collection in:

```text
data/raw/tables_full/
```

The complete collection should contain 7,605 CSV files when counted recursively.

For example, a `.tgz` archive can be extracted with:

```bash
mkdir -p data/raw/tables_full
tar -xzf eurostat_7605_tables.tgz -C data/raw/tables_full
```

The exact internal directory structure of the archive is not important because the pipeline searches for CSV files recursively.

Some source files use the `.csv` extension but contain GZIP-compressed data. The pipeline detects their compression automatically.

The title mapping must be placed in:

```text
data/raw/titles/file-names_to_titles_eurostat_unfiltered.csv
```

The NUTS workbook must be placed in:

```text
data/external/nuts/NUTS2021-NUTS2024.xlsx
```

## Configuration

The pipeline is controlled by:

```text
config/config.yaml
```

### Full-dataset configuration

The final full execution used:

```yaml
dataset_mode: full

paths:
  tables_small: data/raw/tables_small
  tables_full: data/raw/tables_full
  titles: data/raw/titles
  nuts: data/external/nuts
  intermediate: runs/full/intermediate
  processed: runs/full/processed
  outputs: runs/full/outputs
  logs: runs/full/logs

processing:
  sample_limit: null
  chunk_size: 20000
  exclude_files:
    - aact_ali01.csv
```

The excluded file remains part of the original local collection, but the pipeline skips it during file selection.

### Reduced-dataset configuration

For a reduced execution, change both the dataset mode and the run directories:

```yaml
dataset_mode: small

paths:
  tables_small: data/raw/tables_small
  tables_full: data/raw/tables_full
  titles: data/raw/titles
  nuts: data/external/nuts
  intermediate: runs/small/intermediate
  processed: runs/small/processed
  outputs: runs/small/outputs
  logs: runs/small/logs

processing:
  sample_limit: null
  chunk_size: 20000
  exclude_files:
    - aact_ali01.csv
```

Changing only `dataset_mode` without changing the run directories would cause the reduced execution to write into the full-run directories.

### Limited test execution

To process only a fixed number of tables, set:

```yaml
sample_limit: 3
```

For the complete selected collection, use:

```yaml
sample_limit: null
```

## Running the pipeline

Run the complete pipeline from the repository root:

```bash
python main.py
```

The program executes Steps 1–7 sequentially.

For the final full configuration, the initial console output should report:

```text
Excluded files: aact_ali01.csv
Dataset mode: full | Tables: 7604 | Chunk size: 20,000
```

After a successful execution, run the evaluations:

```bash
python evaluate_classification.py
python evaluate_domains.py
```

The evaluation scripts use the existing manually annotated samples. They do not need to rerun the main pipeline.

## Output files

The final full-run results are stored in:

```text
runs/full/outputs/
```

The four principal vocabulary files required by the assignment are:

```text
runs/full/outputs/measures.csv
runs/full/outputs/dimension_names.csv
runs/full/outputs/dimension_values.csv
runs/full/outputs/units.csv
```

Additional generated outputs include:

```text
classification_all.csv
classification_summary.csv
domain_examples.csv
domain_summary.csv
measure_domains.csv
other.csv
```

The complete output directory is preserved in the repository, while large intermediate and processed files are excluded through `.gitignore`.

A reduced execution writes its results to:

```text
runs/small/outputs/
```

provided that the corresponding paths are selected in `config/config.yaml`.

## Processing performance

The final execution used a chunk size of 20,000 rows.

| Stage                        | Measured time |
| ---------------------------- | ------------: |
| Temporal interval extraction |    5 min 46 s |
| Table vocabulary extraction  |   42 min 05 s |

Vocabulary extraction was the most expensive stage because it required reading and decompressing the descriptive content of all 7,604 selected tables.

The later stages operated mainly on smaller intermediate datasets and completed substantially faster.

Execution time depends primarily on:

* CPU performance;
* GZIP decompression speed;
* disk read speed;
* table size;
* selected chunk size.

Internet speed does not affect processing after the required resources have been downloaded.

A sanitised summary of the final execution is stored in:

```text
runs/full/logs/full_run_summary.txt
```

The summary uses relative project paths and does not include private local directory information.

## Error handling

### Incomplete GZIP stream

The file `aact_ali01.csv` was excluded because its internal GZIP content was incomplete.

The file is listed in the `exclude_files` configuration entry, allowing the remaining 7,604 tables to be processed without modifying the original dataset.

### Malformed quotation marks

The following source files contained non-standard quotation-mark patterns:

```text
prc_dap12.csv
prc_dap13.csv
prc_dap14.csv
prc_dap15.csv
```

For these files, quotation marks were treated as ordinary characters during parsing. All four files were successfully processed in the final execution.

### Memory management

Large tables are read in configurable chunks.

The final execution used:

```yaml
chunk_size: 20000
```

This avoids loading the complete collection or the largest individual tables into memory at once.

## Known limitations

* The vocabulary-classification method depends mainly on lexical and structural heuristics.
* The same expression may have different semantic roles in different table contexts.
* Units and dimension values can be difficult to distinguish.
* Residual titles may combine measures, dimensions, populations and qualifiers.
* Geographical recognition is limited to the supplied NUTS resources and supported matching rules.
* Generic measures can be difficult to assign to a specialised statistical domain.
* Domain assignment selects one principal domain per measure.
* Low-confidence domain assignments require greater caution during interpretation.
* The manually annotated evaluation samples cover only a small part of the complete vocabulary.
* The `other` category contains very few automatically classified terms, which limits the reliability of its evaluation metrics.
* The optional semantic-relations stage was not implemented.

## Reproducibility

All paths used by the project are relative to the repository root.

To reproduce the final full execution:

1. Install Python 3.10 or later.
2. Install the packages from `requirements.txt`.
3. Restore the complete table collection under `data/raw/tables_full/`.
4. Restore the title mapping under `data/raw/titles/`.
5. Restore the NUTS workbook under `data/external/nuts/`.
6. Select the full configuration in `config/config.yaml`.
7. Run:

```bash
python main.py
python evaluate_classification.py
python evaluate_domains.py
```

The expected full-run results are:

```text
Selected tables: 7,604
Processing errors: 0
Final vocabulary terms: 162,236
Measures: 11,209
Dimension names: 15,204
Dimension values: 132,992
Units: 2,830
Other terms: 1
Classification accuracy: 0.8642
Domain-assignment accuracy: 0.8588
```

A reduced execution can be used to verify the workflow without processing the complete collection:

1. Place the reduced tables in `data/raw/tables_small/`.
2. Select the small configuration and `runs/small/` paths.
3. Run `python main.py`.

The reduced execution should generate the same types of output files, although its term counts and evaluation coverage will differ from the full run.

## Use of AI tools

AI tools were used as development support for:

* code review;
* debugging;
* investigation of CSV and GZIP parsing errors;
* discussion of implementation alternatives;
* evaluation design;
* documentation drafting.

The production pipeline does not call an external large language model or generative AI service.

Vocabulary classification is based on lexical and structural rules. Domain assignment combines lexical rules with TF-IDF similarity.

All final code changes, manually assigned evaluation labels and reported numerical results were reviewed and validated by the project author.

No AI service, API key or stored prompt is required to install, reproduce or run the project.

## Report

The final report is available at:

```text
report/statistic_vocabularies.pdf
```

It describes the methodology, implementation, experimental results, evaluation, limitations and conclusions of the project.
