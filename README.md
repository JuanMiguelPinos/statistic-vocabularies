# STAR Statistical Vocabularies

Automatic extraction, classification and organisation of statistical vocabulary from Eurostat tables.

## Overview

This project implements a complete pipeline for extracting and organising statistical vocabulary from a large collection of Eurostat tables.

The pipeline processes table headers, descriptive columns and official titles in order to identify temporal expressions, textual terms and geographical entities. It then constructs a global vocabulary, classifies its terms into semantic categories and groups the extracted measures into statistical domains.

The complete execution processed 7,604 tables successfully. One table was excluded because its internal GZIP stream was incomplete.

The optional semantic-relations stage was not implemented.

## Pipeline

For each statistical table (t), the pipeline performs the following steps:

1. Extract temporal expressions and intervals (D(t)).
2. Extract the duplicate-free set (S(t)) of non-numeric strings from the descriptive columns.
3. Identify geographical terms (Geo(t)) using the NUTS nomenclature.
4. Remove geographical terms to obtain `V(t) = S(t) - Geo(t)`.
5. Process the official title and add its residual semantic content to (V(t)).
6. Combine the table vocabularies into a global vocabulary (V).
7. Classify every term as:

   * measure;
   * dimension name;
   * dimension value;
   * unit;
   * other.
8. Assign every measure to one of seventeen statistical domains.
9. Evaluate vocabulary classification and domain assignment using manually annotated samples.

## Final results

The original collection contained 7,605 tables.

The file `aact_ali01.csv` was excluded because its internal GZIP stream ended before the expected end-of-stream marker. The integrity of the original downloaded archive was verified, indicating that the problem was associated with the internal file rather than with an incomplete local download.

| Result                                      |     Value |
| ------------------------------------------- | --------: |
| Original tables                             |     7,605 |
| Excluded tables                             |         1 |
| Selected tables                             |     7,604 |
| Successfully processed tables               |     7,604 |
| Tables with processing errors               |         0 |
| Extracted temporal intervals                |   198,090 |
| Tables without detected dates               |        31 |
| Distinct table-term pairs in (S(t))         | 1,151,463 |
| Distinct provisional vocabulary terms       |   159,477 |
| Geographical table-term pairs               |   413,625 |
| Distinct geographical terms                 |     3,558 |
| Remaining non-geographical table-term pairs |   737,838 |
| Remaining non-geographical terms            |   155,919 |
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

The classification satisfies the consistency check:

```text
11,209 + 15,204 + 132,992 + 2,830 + 1 = 162,236
```

## Measure-domain assignment

All 11,209 measures were assigned to one of seventeen statistical domains.

| Confidence | Assignments | Percentage |
| ---------- | ----------: | ---------: |
| High       |       5,851 |     52.20% |
| Medium     |       2,240 |     19.98% |
| Low        |       3,118 |     27.82% |
| **Total**  |  **11,209** |   **100%** |

The complete distribution is:

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

### Vocabulary classification

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

Run the evaluation with:

```bash
python evaluar_clasificacion.py
```

The generated files are:

```text
evaluation/classification_metrics.csv
evaluation/classification_report.csv
evaluation/classification_confusion_matrix.csv
evaluation/figures/classification_confusion_matrix.png
```

### Domain assignment

The domain sample contained 85 manually labelled measures.

| Metric                |  Value |
| --------------------- | -----: |
| Evaluated measures    |     85 |
| Correct assignments   |     73 |
| Incorrect assignments |     12 |
| Accuracy              | 85.88% |
| Macro F1-score        | 0.8459 |
| Weighted F1-score     | 0.8717 |

Run the evaluation with:

```bash
python evaluar_dominios.py
```

The generated files are:

```text
evaluation/domain_metrics.csv
evaluation/domain_classification_report.csv
evaluation/domain_confusion_matrix.csv
evaluation/figures/domain_confusion_matrix.png
```

The manually annotated samples should not be regenerated during normal use.

The following scripts create new samples and may overwrite the current annotations:

```bash
python preparar_evaluacion.py
python preparar_evaluacion_dominios.py
```

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
├── prompts/
│   ├── domain_assignment.txt
│   └── term_classification.txt
├── report/
│   ├── statistic_vocabularies.tex
│   └── statistic_vocabularies.pdf
├── runs/
│   ├── full/
│   │   ├── logs/
│   │   │   └── full_run_summary.txt
│   │   └── outputs/
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
├── anotar_dominios.py
├── anotar_muestra.py
├── evaluar_clasificacion.py
├── evaluar_dominios.py
├── main.py
├── preparar_evaluacion.py
├── preparar_evaluacion_dominios.py
├── probar_tabla_grande.py
├── requirements.txt
├── .gitignore
└── README.md
```

The original datasets and large intermediate files are excluded from the repository.

## Requirements

The project was developed and tested with Python 3.11.

Main dependencies:

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

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Required data

The original data are not stored in the repository because of their size.

Create the following local structure:

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

Some files use the `.csv` extension but are internally GZIP-compressed. The pipeline detects compression automatically.

## Configuration

The pipeline is controlled by:

```text
config/config.yaml
```

Example full-dataset configuration:

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
  year_min: 1900
  year_max: 2100
  sample_limit: null
  chunk_size: 20000
  exclude_files:
    - aact_ali01.csv
```

For the reduced dataset, change:

```yaml
dataset_mode: small
```

To limit a test execution to a fixed number of tables, use:

```yaml
sample_limit: 3
```

For the complete execution, use:

```yaml
sample_limit: null
```

## Running the pipeline

Run the complete pipeline from the project root:

```bash
python main.py
```

The pipeline executes steps 1 to 7 sequentially.

To test a large table independently:

```bash
python probar_tabla_grande.py
```

## Output files

The complete execution stores its final results in:

```text
runs/full/outputs/
```

The main files are:

```text
classification_all.csv
classification_summary.csv
dimension_names.csv
dimension_values.csv
domain_examples.csv
domain_summary.csv
measure_domains.csv
measures.csv
other.csv
units.csv
```

The reduced execution stores its results in:

```text
runs/small/outputs/
```

## Processing performance

The complete execution used blocks of 20,000 rows.

| Stage                        |        Time |
| ---------------------------- | ----------: |
| Temporal interval extraction |  5 min 09 s |
| Table vocabulary extraction  | 46 min 36 s |

Vocabulary extraction was the most expensive stage because it had to read and decompress the descriptive content of all selected tables.

The later stages processed smaller intermediate files and completed substantially faster.

Execution time mainly depends on:

* CPU performance;
* GZIP decompression speed;
* disk read speed;
* table size;
* selected block size.

Internet speed does not affect processing once the resources have been downloaded.

## Error handling

### Incomplete GZIP stream

The file `aact_ali01.csv` was excluded because its GZIP content was incomplete.

The integrity hash of the original archive was verified successfully, so the issue was not caused by an incomplete local download.

### Malformed quotation marks

The following files contained non-standard quotation marks:

```text
prc_dap12.csv
prc_dap13.csv
prc_dap14.csv
prc_dap15.csv
```

For these four files, quotation marks were treated as ordinary characters during parsing. All four files were successfully processed in the final execution.

## Known limitations

* The classification method depends mainly on lexical and structural rules.
* The same expression may have different roles in different table contexts.
* Units and dimension values can be difficult to distinguish.
* Residual titles may combine measures, dimensions and populations.
* Geographical recognition is limited to the supplied NUTS resources.
* Generic measures can be difficult to assign to a specialised domain.
* Domain assignment uses one main label per measure.
* The manually annotated evaluation samples cover only part of the vocabulary.
* The optional semantic-relations stage was not implemented.

## Reproducibility

All project paths are relative to the repository root.

A reduced execution can be used to verify the complete workflow without processing the full collection:

1. Place the reduced tables in `data/raw/tables_small/`.
2. Set `dataset_mode: small`.
3. Run:

```bash
python main.py
```

The reduced run should generate the complete set of output files.

## Use of AI tools

AI tools were used as development support for:

* code review;
* debugging;
* analysis of CSV and GZIP errors;
* discussion of implementation alternatives;
* evaluation design;
* documentation drafting.

The production pipeline does not call an external LLM or generative AI service. Vocabulary classification is based on lexical and structural rules, while domain assignment combines rules with TF-IDF similarity.

All code modifications, evaluation labels and reported results were reviewed and validated manually.

## Report

The final report and its LaTeX source are available in:

```text
report/statistic_vocabularies.pdf
report/statistic_vocabularies.tex
```

The report describes the methodology, implementation, results, evaluation, limitations and conclusions of the project.

## Project status

* [x] Temporal interval extraction
* [x] Descriptive vocabulary extraction
* [x] Geographical filtering
* [x] Title processing
* [x] Global vocabulary construction
* [x] Vocabulary classification
* [x] Measure-domain assignment
* [x] Manual classification evaluation
* [x] Manual domain evaluation
* [x] Full dataset execution
* [x] Final report
* [ ] Optional semantic relations
