# AI usage

Every use of a generative AI tool in this project, what it was for, and what we changed.

| Date | Tool | What we used it for | What we changed / checked |
| --- | --- | --- | --- |
| 2026-10-24 | Claude Code | Summarised the project brief and profiled the dataset (G3 = 0 rows, G1/G2 correlation) | Re-checked the numbers ourselves in the notebook |
| 2026-10-24 | Claude Code | Compared the five datasets by difficulty; made a learning plan and a working-session checklist (PDFs) | Chose the Student Performance dataset ourselves |
| 2026-10-24 | Claude Code | Generated the empty project skeleton: folders, `pyproject.toml`, `config.yaml`, `.gitignore`, README draft, module placeholders | Created the private GitHub repository and added Chinmay as a collaborator |
| 2026-10-28 | Claude Code | Explained a `NameError` in `data.py` (a docstring closed after the imports) and fixed it with `ruff --fix` |
| 2026-10-28 | Claude Code | Helped in writing the code for `DatasetLoader` (`data.py`) and `build_preprocessor` (`features.py`), with their tests |
| 2026-10-28 | Claude Code | Suggested the cleaning decisions: drop G1/G2 (leakage), drop the 15 students with G3 = 0, stratify the split on pass/fail |
| 2026-10-30 | Claude Code | Helped in writing the code for `Trainer` (`models.py`), `config.py`, `evaluate.py` and `cli.py`, with their tests; suggested the model list, the grids, MAE as the metric and the out-of-fold threshold with a 75 % recall goal |
| 2026-10-30 | Claude Code | Updated the README (usage, results, layout, data credit); gave the Git commands for committing and co-authoring |
| 2026-10-31 | Claude Code | Explained the code and the methods (cross-validation, GridSearchCV, the five models, scaling, one-hot encoding, the pipeline); made a project overview, a code walkthrough of `data.py` and a model explanation (PDFs) | Went through `data.py` line by line together; |
| 2026-10-31 | Claude Code | Drafted the report (`reports/report.pdf`) and its grade histogram |
| 2026-11-01 | Claude Code | Built the slides with deck-kit (`reports/slides.html`, `reports/slides.pdf`) and speaker notes; added the deck-kit credit to the README |
