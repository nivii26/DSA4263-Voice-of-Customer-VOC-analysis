# Basic requirements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
#from sklearn.metrics import make_scorer
from sklearn.metrics import precision_recall_curve
from sklearn import metrics


# For cross validatiion
from sklearn.model_selection import cross_validate
#from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import StratifiedKFold

# For Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

# For logistic regression
from sklearn.linear_model import LogisticRegression

# For non-linear SVM
from sklearn.svm import NuSVC 

# For VADER
import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# for saving models
import joblib

def sa_train_test_split(reviews_csv): 
    '''
    input : processed reviews_csv DataFrame
    output : 2 DataFrames - train_data (70%), test_data(30%)
    function : Splits input data into train and test
    '''

    # Specify shuffle = True, to shuffle the data before splitting to avoid bias due to order
    # Stratify = sentiment, to ensure train and test have same ratio of postive to negative reviews
    train_data, test_data = train_test_split(reviews_csv, test_size=0.3, shuffle=True, stratify=reviews_csv['Sentiment'], random_state=4263)

    train_size = train_data['Sentiment'].value_counts().to_list()
    test_size = test_data['Sentiment'].value_counts().to_list()

    plotdata = pd.DataFrame({"positive":[train_size[0], test_size[0]],
                            "negative":[train_size[1], test_size[1]],
                            }, 
                            index=["Train", "Test"])

    plotdata.plot(kind = "bar", rot = 0, title = 'Class Distribution in Train-Test Split')
    
    return train_data, test_data



def evaluate_model_test(true_sent, predicted_sent, predicted_prob):
    '''
    input : List of true sentiment label, predicted sentiment label, predicted_probability
    output : Print classification_report, scores and confusion matrix, pr_auc curve
    function : Prints classification_report and evaluation metrics on test data
    '''

    print("TEST RESULTS")
    print("Classification Report")
    print(classification_report(true_sent, predicted_sent))

    # Print Score  
    print("PR_AUC score: ", average_precision_score(true_sent, predicted_prob)) # pos_label=0
    
    ## Print Confusion Matrix
    plt.figure(1)
    labels = ['Negative (0)', 'Positive (1)']
    cm = confusion_matrix(true_sent, predicted_sent)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)
    disp.plot(cmap = 'GnBu')
    plt.title('Confusion Matrix')
    plt.show()

    ## Plot PR_AUC Curve
    plt.figure(2)
    precision, recall, _ = precision_recall_curve(true_sent, predicted_prob)

    # create precision-recall curve
    plt.plot(recall, precision, marker='.', label = 'PR_AUC of Model')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR_AUC Curve')
    plt.legend()
    plt.show()



def train_XGB(train_data, test_data):
    '''
    input : 2 DataFrames 'train_data' and 'test_data'
    output : None
    function : Fits XGBoost Classifier on the train_data, and saves model. Evaluate performance on test_data
    '''

    # create a default XGBoost classifier

    model = XGBClassifier(
        random_state=42, 
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
    
    # Define scoring metrics
    score_metrics = ['accuracy', 'f1', 'f1_weighted', 'average_precision']
    
    # create the Kfold object
    num_folds = 5
    kfold = StratifiedKFold(n_splits=num_folds)
    
    # create the grid search object
    n_iter = 1
    grid = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_grid,
        cv=kfold,
        scoring=score_metrics,
        n_jobs=-1,
        n_iter=n_iter,
        refit='f1_weighted',
    )

    # fit grid search
    best_model = grid.fit(train_data.iloc[: , :-1], train_data['Sentiment'])

    # Save cross-validation results
    cv_results = pd.DataFrame(best_model.cv_results_)
    
    # save model
    best_model.best_estimator_.save_model("root/models/sa/xgb_model.json")

    print("CROSS VALIDATION RESULTS XGBoost")
    print("Average Cross Validation score accuracy: ", cv_results['mean_test_accuracy'][0])
    print("Average Cross Validation score F1: ", cv_results['mean_test_f1'][0])
    print("Average Cross Validation score F1_weighted: ", cv_results['mean_test_f1_weighted'][0])
    print("Average Cross Validation score pr_auc: ", cv_results['mean_test_average_precision'][0])
    # print(best_model.best_estimator_)

    # Predict on test data
    xgb_probs = best_model.predict_proba(test_data.iloc[: , :-1])
    xgb_sentiment = best_model.predict(test_data.iloc[: , :-1])
    xgb_probs_df = pd.DataFrame(data = xgb_probs, columns = ['NEGATIVE', 'POSITIVE'])

    # Evalaute goodness of fit of model on test data
    evaluate_model_test(test_data['Sentiment'], xgb_sentiment, xgb_probs_df['POSITIVE'])




def bayes_classifier(train_data, test_data):
    '''
    input : 2 DataFrames 'train_data' and 'test_data'
    output : None
    function : Fits Naive Baye's Classifier on the train_data. Evaluates performance on test_data
    '''

    NB = GaussianNB()
    # create the Kfold object
    kf = StratifiedKFold(n_splits=5)

    score_metrics = ['accuracy', 'f1', 'f1_weighted', 'average_precision']
    cv_results = cross_validate(NB, train_data.iloc[: , :-1], train_data['Sentiment'], cv = kf, scoring = score_metrics)
    
    # Print cross validation scores
    print("CROSS VALIDATION RESULTS")
    print("Average Cross Validation score accuracy :{}".format(cv_results['test_accuracy'].mean()))
    print("Average Cross Validation score F1 :{}".format(cv_results['test_f1'].mean()))
    print("Average Cross Validation score F1_weighted :{}".format(cv_results['test_f1_weighted'].mean()))
    print("Average Cross Validation score pr_auc :{}".format(cv_results['test_average_precision'].mean()))

    # Final fit
    NB.fit(train_data.iloc[: , :-1], train_data['Sentiment'])

    # Predict sentiment labels and probabilities
    NB_pred_sentiment = NB.predict(test_data.iloc[: , :-1])
    NB_pred_prob = NB.predict_proba(test_data.iloc[: , :-1])
    NB_pred_prob_df = pd.DataFrame(data = NB_pred_prob, columns = ['NEGATIVE', 'POSITIVE']) 

    # Evaluate goodness-of-fit of model on test-data
    evaluate_model_test(test_data['Sentiment'], NB_pred_sentiment, NB_pred_prob_df['POSITIVE'])



def logistic_regression(train_data, test_data):
    '''
    input : 2 DataFrames 'train_data' and 'test_data'
    output : None
    function : Fits Logistic Regression Model on the train_data. Evaluates performance on test_data
    '''

    logreg = LogisticRegression()
    # create the Kfold object
    kf = StratifiedKFold(n_splits=5)
    
    score_metrics = ['accuracy', 'f1', 'f1_weighted', 'average_precision']
    cv_results = cross_validate(logreg, train_data.iloc[: , :-1], train_data['Sentiment'], cv = kf, scoring = score_metrics)
    
    # Print cross validation scores
    print("CROSS VALIDATION RESULTS")
    print("Average Cross Validation score accuracy :{}".format(cv_results['test_accuracy'].mean()))
    print("Average Cross Validation score F1 :{}".format(cv_results['test_f1'].mean()))
    print("Average Cross Validation score F1_weighted :{}".format(cv_results['test_f1_weighted'].mean()))
    print("Average Cross Validation score pr_auc :{}".format(cv_results['test_average_precision'].mean()))

    # Final fit
    logreg.fit(train_data.iloc[: , :-1], train_data['Sentiment'])

    # Predict sentiment labels and probabilities
    logreg_pred_sentiment = logreg.predict(test_data.iloc[: , :-1])
    logreg_pred_prob = logreg.predict_proba(test_data.iloc[: , :-1])
    logreg_pred_prob_df = pd.DataFrame(data = logreg_pred_prob, columns = ['NEGATIVE', 'POSITIVE']) 

    # Evaluate goodness-of-fit of model on test-data
    evaluate_model_test(test_data['Sentiment'], logreg_pred_sentiment, logreg_pred_prob_df['POSITIVE'])



def svc_model(train_data, test_data):
    '''
    input : 2 DataFrames 'train_data' and 'test_data'
    output : None
    function : Fits Support Vector Machine (SVM) Classifier on the train_data and saves model. 
               Evaluates performance on test_data
    '''

    svm = NuSVC(gamma="auto", probability = True)
    # create the Kfold object
    kf = StratifiedKFold(n_splits=5)
    
    score_metrics = ['accuracy', 'f1', 'f1_weighted', 'average_precision']
    cv_results = cross_validate(svm, train_data.iloc[: , :-1], train_data['Sentiment'], cv = kf, scoring = score_metrics)
    
    # Print cross validation scores
    print("CROSS VALIDATION RESULTS")
    print("Average Cross Validation score accuracy :{}".format(cv_results['test_accuracy'].mean()))
    print("Average Cross Validation score F1 :{}".format(cv_results['test_f1'].mean()))
    print("Average Cross Validation score F1_weighted :{}".format(cv_results['test_f1_weighted'].mean()))
    print("Average Cross Validation score pr_auc :{}".format(cv_results['test_average_precision'].mean()))

   # Final fit
    svm.fit(train_data.iloc[: , :-1], train_data['Sentiment'])

    # Save model
    joblib.dump(svm, 'root/models/sa/svm_model.pkl')

    # Predict sentiment labels and probabilities
    svm_pred_sentiment = svm.predict(test_data.iloc[: , :-1])
    svm_pred_prob = svm.predict_proba(test_data.iloc[: , :-1])
    svm_pred_prob_df = pd.DataFrame(data = svm_pred_prob, columns = ['NEGATIVE', 'POSITIVE'])

    # Evaluate goodness-of-fit of model on test-data
    evaluate_model_test(test_data['Sentiment'], svm_pred_sentiment, svm_pred_prob_df['POSITIVE'])




def vader(train_data, test_data):
    '''
    input : 2 DataFrames 'train_data' and 'test_data'
    output : None
    function : Evaluates performance of pretrained Vader Model on the train_data and test_data
    '''

    SIA = SentimentIntensityAnalyzer()

    # Performance of Train Data
    vader_train_results = pd.DataFrame()
    vader_train_results['VADER_dict'] = train_data['Text'].apply(lambda text: SIA.polarity_scores(text))
    vader_train_results['VADER_score'] = vader_train_results['VADER_dict'].apply(lambda sent_dict: sent_dict['compound'])
    vader_train_results['VADER_label'] = 0

    # If compound > 0 -> 1 else compund < 0 -> 0
    vader_train_results.loc[vader_train_results['VADER_score'] > 0, 'VADER_label'] = 1
    vader_train_results.loc[vader_train_results['VADER_score'] < 0, 'VADER_label'] = 0
    vader_train_results['VADER_prob'] =  vader_train_results['VADER_dict'].apply(lambda sent_dict: sent_dict['pos'])

    print("TRAIN DATA RESULTS")
    print("Train accuracy :{}".format(accuracy_score(train_data['Sentiment'], vader_train_results['VADER_label'])))
    print("Train F1 :{}".format(f1_score(train_data['Sentiment'], vader_train_results['VADER_label'])))
    print("Train F1_weighted :{}".format(f1_score(train_data['Sentiment'], vader_train_results['VADER_label'], average='weighted')))
    print("Train pr_auc :{}".format(average_precision_score(train_data['Sentiment'], vader_train_results['VADER_prob'])))

    # Performance of Test Data
    vader_results = pd.DataFrame()
    vader_results['VADER_dict'] = test_data['Text'].apply(lambda text: SIA.polarity_scores(text))
    vader_results['VADER_score'] = vader_results['VADER_dict'].apply(lambda sent_dict: sent_dict['compound'])
    vader_results['VADER_label'] = 0

    # If compound > 0 -> 1 else compund < 0 -> 0
    vader_results.loc[vader_results['VADER_score'] > 0, 'VADER_label'] = 1
    vader_results.loc[vader_results['VADER_score'] < 0, 'VADER_label'] = 0

    vader_results['VADER_prob'] =  vader_results['VADER_dict'].apply(lambda sent_dict: sent_dict['pos'])

    evaluate_model_test(test_data['Sentiment'], vader_results['VADER_label'], vader_results['VADER_prob'])



def final_svm_full_model(train_data):
    '''
    input : full preprocessed train data
    output : final svm model
    function : Refit svm model on comeplete training data and save the final model for future unseen predictions
    '''
    
    svm = NuSVC(gamma="auto", probability = True)
    # create the Kfold object
    kf = StratifiedKFold(n_splits=5)
        
    score_metrics = ['accuracy', 'f1', 'f1_weighted', 'average_precision']
    cv_results = cross_validate(svm, train_data.iloc[: , :-1], train_data['Sentiment'], cv = kf, scoring = score_metrics)
        
     # Print cross validation scores
    print("CROSS VALIDATION RESULTS")
    print("Average Cross Validation score accuracy :{}".format(cv_results['test_accuracy'].mean()))
    print("Average Cross Validation score F1 :{}".format(cv_results['test_f1'].mean()))
    print("Average Cross Validation score F1_weighted :{}".format(cv_results['test_f1_weighted'].mean()))
    print("Average Cross Validation score pr_auc :{}".format(cv_results['test_average_precision'].mean()))

    # Final fit
    svm.fit(train_data.iloc[: , :-1], train_data['Sentiment'])

    # Save model
    joblib.dump(svm, 'root/models/sa/final_svm_model.pkl')
    
    if __name__ == "__main__":
        input_data = pd.read_csv(rf"root/src/data/sa/features_train_sa.csv")
        final_svm_full_model(input_data)

    
    
