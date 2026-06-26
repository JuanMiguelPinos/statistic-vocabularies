# STAR Statistical Vocabularies

Automatic extraction, classification and organisation of statistical vocabulary from Eurostat tables.

This project implements a complete pipeline that processes a large collection of statistical tables and transforms their textual content into a structured vocabulary. The system extracts temporal expressions, descriptive terms and geographical entities, builds a global vocabulary, classifies its terms and groups the detected measures into statistical domains.

## Project overview

For every statistical table (t), the pipeline performs the following operations:

1. Extract temporal expressions and intervals (D(t)).
2. Extract the duplicate-free set (S(t)) of non-numeric strings from the descriptive columns.
3. Identify geographical terms (Geo(t)) using the NUTS nomenclature.
4. Remove geographical terms:

[
V(t) = S(t) \setminus Geo(t)
]

5. Process the table title and add its residual semantic content to (V(t)).
6. Construct the global vocabulary:

[
V = \bigcup_t V(t)
]

7. Classify the global vocabulary into:

   * measures;
   * dimension names;
   * dimension values;
   * units;
   * other terms.
8. Assign every detected measure to one of seventeen statistical domains.
9. Evaluate vocabulary classification and domain assignment using manually annotated samples.

The optional semantic-relation stage was not implemented.

---

## Main results

The original collection contained 7,605 tables.

One table, `aact_ali01.csv`, was excluded because its internal GZIP stream was incomplete. The integrity of the downloaded archive was verified, which indicates that the problem was associated with the internal file rather than with the local download.

The final execution processed 7,604 tables successfully.

| Result                                |     Value |
| ------------------------------------- | --------: |
| Original tables                       |     7,605 |
| Excluded tables                       |         1 |
| Selected tables                       |     7,604 |
| Successfully processed tables         |     7,604 |
| Tables with errors                    |         0 |
| Extracted temporal intervals          |   198,090 |
| Distinct table-term pairs in (S(t))   | 1,151,463 |
| Distinct provisional vocabulary terms |   159,477 |
| Geographical table-term pairs         |   413,625 |
| Distinct geographical terms           |     3,558 |
| Final table-term pairs                |   745,418 |
| Distinct terms in (V)                 |   162,236 |

### Vocabulary classification

| Category         |   Terms | Percentage |
| ---------------- | ------: | ---------: |
| Measures         |  11,209 |      6.91% |
| Dimension names  |  15,204 |      9.37% |
| Dimension values | 132,992 |     81.97% |
| Units            |   2,830 |      1.74% |
| Other            |       1 |     <0.01% |
| Total            | 162,236 |       100% |

The classification totals satisfy:

[
11,209 + 15,204 + 132,992 + 2,830 + 1 = 162,236
]

### Domain assignment

The 11,209 measures were assigned to seventeen statistical domains.

| Confidence level | Assignments | Percentage |
| ---------------- | ----------: | ---------: |
| High             |       5,851 |     52.20% |
| Medium           |       2,240 |     19.98% |
| Low              |       3,118 |     27.82% |
| Total            |      11,209 |       100% |

The largest groups were:

| Domain                                  | Measures |
| --------------------------------------- | -------: |
| Other or multidisciplinary              |    2,844 |
| Labour market                           |    1,475 |
| Science, technology and digital society |    1,266 |
| Education and training                  |    1,042 |
| Business, industry and trade            |      879 |

---

## Evaluation

Two manually annotated samples were used to evaluate the semantic stages.

### Vocabulary classification evaluation

The evaluation sample contained 81 manually labelled terms.

| Metric                |  Value |
| --------------------- | -----: |
| Evaluated terms       |     81 |
| Correct predictions   |     70 |
| Incorrect predictions |     11 |
| Accuracy              | 86.42% |

Per-category results:

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

Generated files:

```text
evaluation/classification_metrics.csv
evaluation/classification_report.csv
evaluation/classification_confusion_matrix.csv
evaluation/figures/classification_confusion_matrix.png
```

### Domain-assignment evaluation

The domain evaluation sample contained 85 manually annotated measures.

| Metric                |  Value |
| --------------------- | -----: |
| Evaluated measures    |     85 |
| Correct predictions   |     73 |
| Incorrect predictions |     12 |
| Accuracy              | 85.88% |
| Macro F1-score        | 0.8459 |
| Weighted F1-score     | 0.8717 |

Run the evaluation with:

```bash
python evaluar_dominios.py
```

Generated files:

```text
evaluation/domain_metrics.csv
evaluation/domain_classification_report.csv
evaluation/domain_confusion_matrix.csv
evaluation/figures/domain_confusion_matrix.png
```

The manually annotated samples must not be regenerated unless a new evaluation is intentionally required.

Do not run:

```bash
python preparar_evaluacion.py
python preparar_evaluacion_dominios.py
```

These scripts may overwrite the existing manual annotations.

---

## Repository structure

```text
statistic-vocabularies/
├── config/
│   └── config.yaml
├── data/
│   ├── external/
│   │   └── nuts/
│   ├── raw/
│   │   ├── tables_small/
│   │   ├── tables_full/
│   │   └── titles/
│   ├── intermediate/
│   └── processed/
├── evaluation/
│   ├── figures/
│   ├── classification_metrics.csv
│   ├── classification_report.csv
│   ├── classification_confusion_matrix.csv
│   ├── domain_metrics.csv
│   ├── domain_classification_report.csv
│   ├── domain_confusion_matrix.csv
│   ├── gold_sample.csv
│   └── domain_gold_sample.csv
├── runs/
│   ├── small/
│   │   ├── intermediate/
│   │   ├── outputs/
│   │   └── logs/
│   └── full/
│       ├── intermediate/
│       ├── outputs/
│       └── logs/
├── src/
│   ├── classify_terms.py
│   ├── cluster_measures.py
│   ├── config.py
│   ├── evaluate.py
│   ├── extract_geo.py
│   ├── extract_time.py
│   ├── extract_vocabulary.py
│   ├── io_utils.py
│   ├── pipeline.py
│   ├── process_titles.py
│   └── semantic_relations.py
├── main.py
├── evaluar_clasificacion.py
├── evaluar_dominios.py
├── probar_tabla_grande.py
├── requirements.txt
├── .gitignore
├── README.md
└── report.pdf
```

The original statistical tables and the large intermediate files are not included in the repository.

---

## Requirements

The project was developed with:

* Python 3.11
* pandas
* NumPy
* scikit-learn
* openpyxl
* PyYAML
* tqdm
* matplotlib

A solid-state drive is recommended for processing the complete collection.

The extracted full dataset occupies approximately 100 GB. Internet access is not required after the resources have been downloaded.

---

## Installation

### 1. Clone the repository

```bash
git clone [PUBLIC_REPOSITORY_URL]
cd statistic-vocabularies
```

### 2. Create a Conda environment

```bash
conda create -n star_vocab python=3.11
conda activate star_vocab
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

---

## Required data

The datasets are not included because of their size.

### Statistical tables

Place the reduced table collection in:

```text
data/raw/tables_small/
```

Place the complete collection in:

```text
data/raw/tables_full/
```

Some files use the `.csv` extension but contain internally GZIP-compressed data. Compression is detected automatically by the pipeline.

### Table-title mapping

Place the title file in:

```text
data/raw/titles/
```

Expected filename:

```text
file-names_to_titles_eurostat_unfiltered.csv
```

### NUTS nomenclature

Place the geographical workbook in:

```text
data/external/nuts/
```

Expected filename:

```text
NUTS2021-NUTS2024.xlsx
```

---

## Configuration

The pipeline is configured through:

```text
config/config.yaml
```

Example configuration for the complete dataset:

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
  use_cache: true
  sample_limit: null
  random_seed: 42
  chunk_size: 20000
  exclude_files:
    - aact_ali01.csv

classification:
  allow_other: true
  minimum_confidence: 0.70

clustering:
  number_of_domains: 17
  method: tfidf_rules
```

### Reduced execution

Use:

```yaml
dataset_mode: small
```

The reduced dataset is recommended for development and reproducibility tests.

### Full execution

Use:

```yaml
dataset_mode: full
```

and:

```yaml
sample_limit: null
```

### Limited test

To process only a small number of files:

```yaml
sample_limit: 3
```

### Chunk size

Large tables are read incrementally:

```yaml
chunk_size: 20000
```

A smaller block size reduces peak memory usage but may increase execution time.

---

## Running the pipeline

From the repository root, run:

```bash
python main.py
```

The program executes steps 1 to 7 sequentially.

A complete execution generates intermediate files and final outputs in the directories configured in `config.yaml`.

To test a large table independently:

```bash
python probar_tabla_grande.py
```

---

## Pipeline stages

### Step 1: temporal extraction

The table headers are analysed to detect individual years and temporal ranges.

Outputs:

```text
table_dates.csv
table_dates_summary.csv
```

### Step 2: extraction of (S(t))

The pipeline identifies the leftmost descriptive columns and extracts their distinct non-numeric strings.

Large tables are processed in blocks, and only the required descriptive columns are loaded.

Outputs:

```text
table_strings.csv
global_table_vocabulary.csv
table_strings_summary.csv
```

### Step 3: geographical filtering

A geographical dictionary is created from the NUTS workbook.

Terms matching geographical names, codes or aliases are stored separately and removed from the provisional vocabulary.

Outputs:

```text
geography_dictionary.csv
table_geographies.csv
table_vocabulary_without_geo.csv
```

### Steps 4 and 5: title processing and global vocabulary

Every table is associated with its official title.

Dates and geographical references are removed from the title, and the remaining semantic expression is incorporated into the table vocabulary.

Outputs:

```text
title_processing.csv
global_vocabulary.csv
```

### Step 6: vocabulary classification

Every term is assigned to exactly one category:

* measure;
* dimension name;
* dimension value;
* unit;
* other.

Outputs:

```text
classification_all.csv
measures.csv
dimension_names.csv
dimension_values.csv
units.csv
other.csv
```

### Step 7: domain assignment

Measures are assigned to one of seventeen statistical domains using high-precision lexical rules and TF-IDF similarity.

Outputs:

```text
measure_domains.csv
domain_summary.csv
domain_examples.csv
```

---

## Final output files

The main final files are located in:

```text
runs/full/outputs/
```

Expected files:

```text
classification_all.csv
measures.csv
dimension_names.csv
dimension_values.csv
units.csv
other.csv
measure_domains.csv
domain_summary.csv
domain_examples.csv
```

Intermediate evidence is generated in:

```text
runs/full/intermediate/
```

These intermediate files can be very large and are therefore excluded from version control.

---

## Processing performance

The complete execution was performed locally with a block size of 20,000 rows.

| Stage                 |        Time |
| --------------------- | ----------: |
| Temporal extraction   |  5 min 09 s |
| Vocabulary extraction | 46 min 36 s |

Vocabulary extraction was the most expensive stage because it required reading and decompressing the descriptive content of all selected tables.

The later stages worked mainly with smaller intermediate files and completed substantially faster.

Execution time depends mainly on:

* CPU performance;
* GZIP decompression speed;
* disk read speed;
* source-file size;
* selected block size.

Internet speed does not affect the pipeline once the datasets have been downloaded.

---

## Error handling

### Incomplete GZIP file

The file:

```text
aact_ali01.csv
```

was excluded because its internal GZIP stream ended before the expected end-of-stream marker.

The original archive hash was verified successfully, so the problem was not caused by an incomplete local download.

### Malformed quotation marks

The following files contained non-standard quotation marks:

```text
prc_dap12.csv
prc_dap13.csv
prc_dap14.csv
prc_dap15.csv
```

For these files, quotation marks were treated as ordinary characters during vocabulary extraction. All four files were processed successfully in the final execution.

---

## Known limitations

* The classification approach is mainly rule-based and depends on lexical evidence.
* The same term may have different semantic roles in different tables.
* Units and dimension values can be difficult to distinguish.
* Complete title residuals may combine measures and dimensions.
* Geographical recognition is limited to the supplied NUTS resources.
* Generic measures are difficult to assign to a specialised domain.
* Domain assignment is single-label even when a measure may belong to several domains.
* The manual evaluation samples represent only a small subset of the complete vocabulary.
* The optional semantic-relation stage was not implemented.

---

## Reproducibility

All project paths are relative to the repository root.

The random seed is fixed to:

```yaml
random_seed: 42
```

To verify reproducibility without processing the complete dataset:

1. Install the required environment.
2. Place the reduced dataset in `data/raw/tables_small/`.
3. Set:

```yaml
dataset_mode: small
```

4. Run:

```bash
python main.py
```

The reduced execution should generate the complete set of intermediate and final files.

---

## Use of AI tools

AI tools were used as support for:

* code review;
* debugging;
* documentation;
* discussion of implementation alternatives;
* design of the evaluation process;
* analysis of parsing and compression errors.

All proposed modifications were reviewed, executed and validated manually.

The final classifications, evaluation samples and reported results were inspected using the generated project outputs.

---

## Report

The final report is available as:

```text
report.pdf
```

It describes:

* input resources;
* project architecture;
* temporal extraction;
* vocabulary extraction;
* geographical recognition;
* title processing;
* vocabulary classification;
* domain assignment;
* evaluation;
* implementation decisions;
* limitations;
* conclusions.

---

## Repository

Public repository:

```text
[PUBLIC_REPOSITORY_URL]
```

---

## Project status

* [x] Temporal extraction
* [x] Descriptive vocabulary extraction
* [x] Geographical filtering
* [x] Title processing
* [x] Global vocabulary construction
* [x] Vocabulary classification
* [x] Measure-domain assignment
* [x] Manual vocabulary evaluation
* [x] Manual domain evaluation
* [x] Full dataset execution
* [x] Report
* [ ] Optional semantic relations
