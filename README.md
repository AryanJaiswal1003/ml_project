# Student Exam Performance Prediction

This project predicts a student's mathematics score from exam records and serves that prediction through a Flask application. Training reads `notebook/data/stud.csv`, compares several regression models, and saves the selected model together with the preprocessing steps used to prepare its inputs.

## Dataset

The source file is `notebook/data/stud.csv`. Each row is one student. The columns are:

| Column | Role |
| --- | --- |
| `gender` | Input |
| `race_ethnicity` | Input |
| `parental_level_of_education` | Input |
| `lunch` | Input |
| `test_preparation_course` | Input |
| `reading_score` | Input |
| `writing_score` | Input |
| `math_score` | Target |

Training uses an 80/20 split with `random_state=42`. The split is written to `artifacts/train.csv` and `artifacts/test.csv`. A copy of the full table is written to `artifacts/data.csv`.

## Project structure

```
ML-PROJECT/
├── app.py
├── requirements.txt
├── setup.py
├── .ebextensions/
│   └── python.config
├── artifacts/
├── logs/
├── notebook/
│   ├── data/stud.csv
│   ├── EDA-Analysis.ipynb
│   └── Model-Training.ipynb
├── src/
│   ├── exception.py
│   ├── logger.py
│   ├── utils.py
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   └── pipeline/
│       ├── train_pipeline.py
│       └── predict_pipeline.py
└── templates/
    ├── index.html
    └── home.html
```

`notebook/EDA-Analysis.ipynb` and `notebook/Model-Training.ipynb` are the exploratory and model-comparison notebooks. The package under `src/` is the training and prediction code used by the application.

## Training

Run the training entry point from the project root:

```powershell
python src/pipeline/train_pipeline.py
```

`src/pipeline/train_pipeline.py` calls the three components in order.

1. `DataIngestion` reads `notebook/data/stud.csv` and writes the raw, training, and test CSV files under `artifacts/`.
2. `DataTransformation` prepares the feature columns and saves the fitted preprocessor to `artifacts/preprocessor.pkl`. Numeric columns (`reading_score`, `writing_score`) are filled with the training median and scaled. Categorical columns are filled with the most frequent training value, one-hot encoded, and scaled. `math_score` is held out and appended as the last column of each array.
3. `ModelTrainer` fits the candidate models with `GridSearchCV` (`cv=3`), selects the model with the highest test R², and saves it to `artifacts/model.pkl`. A model is saved only when that test R² is at least 0.6. The command prints the test R² of the saved model.

The candidates are Random Forest, Decision Tree, Gradient Boosting, Linear Regression, K-Neighbour Regressor, XGBRegressor, CatBoosting Regressor, and AdaBoost Regressor. Parameter grids are defined in `src/components/model_trainer.py`. `src/utils.py` saves and loads the preprocessor and model with `dill`.

The same three steps can also be started from `src/components/data_ingestion.py`. `src/pipeline/train_pipeline.py` is the training entry point.

Logs for each run are written to a timestamped file in `logs/`. Failures are raised as `CustomException` from `src/exception.py`.

## Prediction

`PredictPipeline` in `src/pipeline/predict_pipeline.py` loads `artifacts/preprocessor.pkl` and `artifacts/model.pkl`. It applies the saved preprocessor to a one-row frame and returns the model's mathematics-score prediction. Both artifact files must exist before the application can predict, so the training command has to be run first.

The Flask application in `app.py` exposes two routes:

| Route | Method | Result |
| --- | --- | --- |
| `/` | GET | Renders `templates/index.html` |
| `/predict` | GET | Renders the input form in `templates/home.html` |
| `/predict` | POST | Returns the predicted mathematics score on the same page |

The form collects gender, ethnicity, parental level of education, lunch, test preparation course, reading score, and writing score.

## Installation

The package is developed with Python 3.8. Commands below are run from the project root. The training and prediction modules import `src`, so the project is installed in editable mode.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` installs pandas, NumPy, seaborn, Matplotlib, scikit-learn, CatBoost, XGBoost, and dill. The web application also requires Flask, which is imported by `app.py` and is not listed in `requirements.txt`:

```powershell
pip install flask
```

## Run the application

Start the server from the project root after the artifacts have been created:

```powershell
python app.py
```

The application listens on `0.0.0.0` at port `5000`. Open `http://127.0.0.1:5000/` for the welcome page and `http://127.0.0.1:5000/predict` for the prediction form.

## Deployment

`.ebextensions/python.config` sets the Elastic Beanstalk Python WSGI path to `app:app`. That value imports the `app` object from `app.py`. The deployed bundle must include the trained `artifacts/preprocessor.pkl` and `artifacts/model.pkl` files, because prediction loads them at request time.
