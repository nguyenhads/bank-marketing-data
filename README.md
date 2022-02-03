# BANK MARKETING - DATA ANALYSIS PROJECT



## Project Description

    - This project is to analyze a data of a marketing campaign provided by banking institutions (can be considered as our customer). 
    - The requirement of the bank institutions is to promote their clients to subscribe their services (in this case: term deposit).

## Goals of this project:

    1. Build a machine learning model that are able to predict if a client will subscribe to the banks' service 
    
    2. Apply various useful and advanced machine learning concepts to a real data analysis project. The main concepts were listed following:
           - Testing - Unit test in python (テスト入門)
           - Scikit-learn Intermediate: customized transformer, pipeline to automatically pre-process data (scikit-learn中級編)
           - Pre-processing data (データ分析の前処理)
           - Interpretation of forecast results using SHAP values (SHAP値を用いた予測結果の解釈)

### Data source:
    

- UCI Machine Learning Repository - [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#)


### Construction of this project

- `test_data.py` (customized_packages/unit_test/test_data.py) : apply unit test to confirm the downloaded data
- `01_exploratory_data_analysis.ipynb` : Overview of data and visualization
- `02_preprocess_data.ipynb` : Preprocess data using Scikit-learn intermidate concepts (customized transformer, Pipeline)
- `03_build_model.ipynb`: Build models using [Random Forrest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and [XGBoost](https://xgboost.readthedocs.io/en/latest/index.html)
- `04_model_evaluation.ipynb`: Interpretation of model' results using [SHAP library](https://github.com/slundberg/shap)
