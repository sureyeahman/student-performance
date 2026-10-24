# Student Performance: final grade predictor

A command-line tool that predicts a student's final Portuguese grade (G3, 0–20) at the
start of the school year, so a school counsellor can flag students at risk of failing
(predicted G3 below 10) and offer support early.

Final project for Machine Learning × Advanced Python (PGE 2, 2026).
Team: Shauryaman Singh, Chinmay Hajare.

## Install

Requires Python 3.11 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
student-performance train                              # train and save the model
student-performance evaluate                           # print the test metrics
student-performance predict --input new_case.json      # predict one new student
```

## Tests and linting

```bash
pytest
ruff check .
```

## Project layout

```
config.yaml               paths, seed, hyperparameter grids
data/raw/                 the dataset (student-por.csv) and its column descriptions
src/student_performance/
    data.py               DatasetLoader: load, clean, split
    features.py           ColumnTransformer and custom features
    models.py             Trainer: pipelines, CV, save, load
    evaluate.py           metrics, decision rule, plots
    cli.py                train, evaluate, predict
tests/                    pytest tests
notebooks/                analysis notebook (calls the package)
models/                   saved model (created by `train`)
reports/                  report and figures
```

## Data source

Student Performance dataset, UCI Machine Learning Repository, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):
<https://archive.ics.uci.edu/dataset/320/student+performance>

P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student
Performance.* In Proceedings of the 5th FUture BUsiness TEChnology Conference
(FUBUTEC 2008), pp. 5–12, Porto, Portugal, 2008.

We use the Portuguese course file (`student-por.csv`, 649 students).

## Sources and credits

See `AI_USAGE.md` for the generative AI tools we used.
