# Business License Status Prediction

- The problem statement was presented at [ZS Data Science Challenge - 2019](https://www.interviewbit.com/contest/zs-yds-2019/)
- [Dataset Link](https://www.kaggle.com/datasets/samacker77k/classification-of-business-license-status)
- [Problem Statement & Feature Description](https://github.com/avr2002/Business-License-Status-Prediction/blob/main/notebooks/feature_description.md)


## 1. Create a new environment

   - `conda create -p venv python==3.10 -y`
   - `conda activate venv/`

## 2. Install all the requirements

- `python -m pip install --upgrade pip`

- `pip install -r requirements.txt`

- `conda install jupyter` (to run the jupyter notebook)

## 3. Code Execution

   - Run `python src/engine.py` to train/predict/deploy

   
* **

## Aim
- To introduce the Deep Neural Network and its implementation

- The project aims to predict if a customer's license should be issued, renewed, or cancelled depending on various parameters in [Business License Dataset](https://www.kaggle.com/datasets/samacker77k/classification-of-business-license-status)

## Data Description
The dataset used is a licensed dataset. It contains information about 86K different businesses over various features. The target variable is the status of license which has five different categories.

## Tech Stack
- ➔ Language: `Python`
- ➔ Libraries: `pandas`, `seaborn`, `numpy`, `matplotlib`, `scikit-learn`, `h2o`, `tensorflow`, `flask`, `gunicorn`

## Approach
1. Data Description
2. Exploratory Data Analysis
3. Data Cleaning & Feature Engineering
4. Preparing data for analysis
5. Base model(Random Forest) building using `h2o`
6. Building deep neural network model
7. Predictions on test data
8. Model deployment using `flask` `gunicorn`


## Folder Structure
```
input
    |__License_Data.csv
    |__preprocessed_License_Data.csv (saved the preprocessed data)
    |__test_data.csv

notebooks
    |__utils
        |__helper_functions.py
    |__Features_Description.ipynb
    |__model_api.ipynb (for testing model deployment)
    |__model_notebook.ipynb (Main Notebook)

output
    |__saved_models
        |__h2o_models
        |__neural_network_models

src
    |__ML_Pipeline
        |__`__init__.py`
        |__constants.py
        |__deploy.py
        |__predict.py
        |__preprocessing.py
        |__save_and_load_model.py
        |__train_model.py
        |__wsgi.py
        |__wsgi.sh
    |__`__init__.py`
    |__engine.py

requirements.txt
```


## Project Takeaways
1. What is a Deep Neural Network?
2. Building blocks of Deep Neural Network
3. What is the Activation Function?
4. What is Feedforward?
5. What is Backpropagation?
6. Loss function and its examples
7. What is Dropout regularization?
8. Deep learning libraries such as Tensorflow, Pytorch, Pytorch lightning, Horovod
9. Understanding the Business context and objective
10. Data Cleaning
11. How to prepare data for modeling?
12. How to use the h2o framework for baseline modeling?
13. How to build a Deep Neural Network Model?
14. Hyperparameter tuning
15. Model predictions on test data
16. Model deployment using flask gunicorn
17. Predictions using the deployed model on the server


#### Helpful Links
- [Relative-Paths](https://towardsdatascience.com/simple-trick-to-work-with-relative-paths-in-python-c072cdc9acb9)