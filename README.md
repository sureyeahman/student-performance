# Student Performance: final grade predictor

A command-line tool that predicts a student's final Portuguese grade (G3, 0–20) at the
start of the school year, so a school counsellor can flag students at risk of failing
and offer support early. Grades G1 and G2 are not used: they are not known yet at
that point.

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

Run from the project folder (the commands read `config.yaml`):

```bash
student-performance train                              # tune 5 models with CV, save the best
student-performance evaluate                           # test-set metrics + residual plot
student-performance predict --input new_case.json      # predict one new student
```

- `train` tunes Ridge, Lasso, k-NN, linear SVR and Huber regression with `GridSearchCV`
  (5-fold CV on the training set), keeps the model with the lowest MAE, chooses the
  flag threshold on out-of-fold predictions and saves everything to `models/model.joblib`.
- `evaluate` scores the saved model once on the test set and saves
  `reports/figures/residuals.png`.
- `predict` reads a JSON file with the 30 input columns of one student (or a list of
  students), e.g. `new_case.json`, and prints the predicted grade and the flag:

```
Predicted final grade: 7.7 / 20 -> AT RISK (threshold 11.3)
```

Another config file can be given with `student-performance --config path/to/config.yaml ...`.

## Results

| Model (5-fold CV on the training set) | MAE |
| --- | --- |
| Dummy (predicts the mean) | 2.26 |
| Linear regression | 1.85 |
| Huber regression (chosen) | 1.83 |

Test set (127 students, used once): **MAE 1.66**, RMSE 2.12. A student is flagged when
the predicted grade is below 11.3, which catches 75 % of failing students in
cross-validation (65 % on the test set).

## Tests and linting

```bash
pytest
ruff check .
```

## Project layout

```
config.yaml               paths, seed, CV, decision goal, hyperparameter grids
new_case.json             example input for `predict`
data/raw/                 the dataset (student-por.csv) and its column descriptions
src/student_performance/
    config.py             reads config.yaml
    data.py               DatasetLoader: load, clean, split
    features.py           ColumnTransformer (scaling + one-hot encoding)
    models.py             Trainer: pipelines, CV, tuning, save, load
    evaluate.py           metrics, flag rule and threshold, coefficients, plots
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

- Slides (`reports/slides.html`, `reports/slides.pdf`) built with
  [deck-kit](https://github.com/kiselas/every-slide-is-code) (MIT licence).
- See `AI_USAGE.md` for the generative AI tools we used.
