# Ranking Grid Cells for Landmine Survey in Antioquia, Colombia

Supervised machine learning in Python on the RELand landmine-risk dataset (26,761 grid cells of 500 × 500 m in 15 municipalities of Antioquia, Colombia): a logistic regression and gradient-boosted trees that rank a municipality's cells for survey, tested on municipalities they had never seen.

I chose this problem because, in my experience in humanitarian mine action, survey is the critical stage that allows us to commit the in-depth resources required to complete the task. Survey is a quicker task, like a reconnaissance, and it ensures that the proper resources, from heavy and specialised equipment to teams on site for long periods, go to the correct areas. My goal is a digital survey: an information tool that helps to plan even the surveys more efficiently. This is the third project of my AI capstone on humanitarian mine action.

## Result
Tested on each municipality in turn after training on the other 14, neither model beat a simple rule that surveys first the cells nearest a recorded past landmine event. Over the 12 municipalities with at least five mine cells, the rule's mean PR-AUC was .215, against .183 for the logistic regression and .189 for the trees; over all 15 the models' means were higher (.213 and .218, against .194), but the difference came from La Unión, whose single mine cell both models ranked first. Surveying the top 10% of each municipality's cells would have found 61.6% of all mine cells with the rule, 59.1% with the logistic regression and 53.5% with the trees; without the distance to past events, the models found 23.3% and 17.0%. Testing on cells picked at random instead of whole municipalities would have about doubled the trees' score.

## What's in this repository
- `modeling.ipynb`: the problem and data, loading and checks, exploration, preprocessing, the evaluation design, training and testing on held-out municipalities, the results, a check with one random split, the logistic regression's weights, and a summary.
- `landmine_detection.parquet`: the dataset's data file, unchanged.
- `figures/`: the six charts, written by the notebook.
- `Machine_Learning_Analysis_Report.pdf` and `module_summary.pdf`: the report. The two files are identical; the brief uses both names.
- `requirements.txt`: the Python environment, from `pip freeze`.

## Dataset
*Landmine Detection in Antioquia, Colombia*, by mdulcer10 on Hugging Face: https://huggingface.co/datasets/mdulcer10/landmine-detection. It was published with the RELand study of landmine risk in Colombia (Dulce Rubio et al., 2024, https://doi.org/10.1145/3648437).

- Snapshot: `landmine_detection.parquet`, the dataset's only data file (`data/train-00000-of-00001.parquet` at revision `30710e43dc447ee95bce30e9ed264f68e454e00c`), downloaded on 2026-10-18; SHA-256 `e18e6a278cc62336ff13a68342f5d8a848aebb5e4b35e0cc64dc654160090337`. The notebook checks the checksum when it loads the file.
- Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The Parquet file is shared here unchanged under that licence.
- Source: Hugging Face is not one of the sources the project brief lists. I used this dataset because it is the published data of a peer-reviewed study, with a licence and a recommended testing protocol, and its labels come from the records of Colombia's national mine action authority (AICMA).
- Coordinates: the file gives each cell's latitude and longitude. The notebook uses them once, to check for repeated cells and to count neighbouring mine cells, and then drops them; no model, chart or table uses them.
- Use: the dataset card warns that predictions should not be "the sole basis for demining operations without expert human review and validation in the field". These models should not be used on their own, only as an efficiency aid that orders survey, and no land should be cancelled or released on a model score. Before a model set the order of survey in the field, it would have to show a reasonable confidence level and then be tried for a period, demonstrating that deployment teams were actually more efficient.

## How to run
Needs Python 3.13 and Git. `requirements.txt` was generated on Python 3.13.0.

```
git clone https://github.com/travisrcarlson/applied-machine-learning-project.git
cd applied-machine-learning-project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook modeling.ipynb
```

On macOS or Linux, activate with `source .venv/bin/activate`. If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. In Jupyter, choose **Kernel > Restart Kernel and Run All Cells**. A full run took about 11 minutes on a two-core machine; most of that time is the training of the gradient-boosted trees, and the notebook prints the time each set of 15 rounds took.
