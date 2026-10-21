# Data

The SSD-2026 dataset is distributed by the shared-task organisers on [Codabench](https://www.codabench.org/competitions/13598/) and is not included in this repository. Download it there and follow the organisers' terms of use.

## Expected layout

```
data/
├── en/  train.csv  dev.csv  test.csv
└── es/  train.csv  dev.csv  test.csv
```

`scripts/run_all.sh` looks for per-task files first (`task1_train.csv`, …) and falls back to `train.csv`, so either one file per subtask or one shared file works.

## Columns

Each file needs a text column and a label column. Defaults are `text` and `label`; override them with `--text-col` and `--label-col`.

| Task | Accepted labels |
|---|---|
| 1 | `Support`, `Non-Support` (also `Not Support`, `0`/`1`) |
| 2 | `Individual`, `Group` |
| 3 | `Nation`, `LGBTQ`, `Black Community`, `Religion`, `Women`, `Other` |

Rows with an empty label are skipped, so a single hierarchical file — where Task 2 and Task 3 labels are only filled in for supportive comments — can be used for every subtask by pointing `--label-col` at the right column.

## Samples

`sample_task1.csv`, `sample_task2.csv` and `sample_task3.csv` contain a few **invented** comments in this format so the code and tests can run without the real data. They are not taken from SSD-2026.
