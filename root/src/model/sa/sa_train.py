# Basic requirements
import pandas as pd
import numpy as np

# Train - test split
from sklearn.model_selection import train_test_split

# For XGBoost
from xgboost import XGBClassifier
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

# For flair
from flair.nn import Classifier
from flair.data import Sentence

# For metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold

def sa_train_test_split(reviews_csv): 
    '''
    input : rawdata processed reviews_csv
    output : 2 DataFrames - train_data, test_data
    '''
    positive_reviews = reviews_csv[reviews_csv.Sentiment == 'positive']
    negative_reviews = reviews_csv[reviews_csv.Sentiment == 'negative']

    # split data by 3:7
    positive_train, positive_test = train_test_split(positive_reviews, test_size=0.3, random_state=101)
    negative_train, negative_test = train_test_split(negative_reviews, test_size=0.3, random_state=101)

    print("No. of positive training examples: ", positive_train.shape)
    print("No. of positive testing examples: ", positive_test.shape)
    print("No. of negative training data: ", negative_train.shape)
    print("No. of negative testing examples: ", negative_test.shape)

    train_data = pd.concat([positive_train, negative_train])
    test_data = pd.concat([positive_test, negative_test])

    print("Total no. of training examples: ", train_data.shape)
    print("Total no. of testing examples: ", test_data.shape)

    return train_data, test_data

# def bayes_classifier(train, test):

def train_XGB(train_data):
    # create a default XGBoost classifier

    model = XGBClassifier(
        random_state=42, 
        eval_metric=["error", "auc"]
    )
    # Create the grid search parameter grid and scoring funcitons
    param_grid = {
        "learning_rate": [0.1],
        "colsample_bytree": [0.8],
        "subsample": [0.6],
        "max_depth": [3],
        "n_estimators": [400],
        "reg_lambda": [1],
        "gamma": [0.1],
    }
    scoring = {
        'AUC': 'roc_auc', 
        'Accuracy': make_scorer(accuracy_score)
    }
    # create the Kfold object
    num_folds = 3
    kfold = StratifiedKFold(n_splits=num_folds)
    # create the grid search object
    n_iter=50
    grid = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_grid,
        cv=kfold,
        scoring=scoring,
        n_jobs=-1,
        n_iter=n_iter,
        refit="AUC",
    )

    label_map = {
        'positive': 1,
        'negative': 0,
    }

    # fit grid search
    best_model = grid.fit(train_data.iloc[: , :-1], train_data['Sentiment'].map(label_map))

    print(f'Best training score: {best_model.best_score_}')

    # save model
    best_model.best_estimator_.save_model("models/sa/xgb_model_final.json")
    print("XGB Model has been trained")
    print(best_model.best_estimator_)


def evaluate_model_test(true_sent, predicted_sent, predicted_prob):
    '''
    input : List of true sentiment label, predicted sentiment label, predicted_probability
    output : Print scores and confusion matrix
    '''

    # Print Scores
    print("F1 score: ", f1_score(true_sent, predicted_sent ))
    print("PR_AUC score: ", average_precision_score(true_sent, predicted_prob))
    print("ROC_AUC score: ", roc_auc_score(true_sent, predicted_prob))
    print("Accuracy: ", accuracy_score(true_sent, predicted_sent))    
    
    # Print Confusion Matrix
    labels = ['Negative (0)', 'Positive (1)']
    cm = confusion_matrix(true_sent, predicted_sent)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)
    disp.plot(cmap = 'YlGnBu'); 


