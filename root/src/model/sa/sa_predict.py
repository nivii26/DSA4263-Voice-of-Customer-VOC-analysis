# Basic requirements
import pandas as pd
import numpy as np

# For XGBoost
import xgboost as xgb
from xgboost import XGBClassifier

# For svm
from sklearn.svm import NuSVC 

# For flair
from flair.nn import Classifier
from flair.data import Sentence
import joblib


def svm_predict(test_data, mode):

    # Load the trained SVM model
    if mode == 'train':
        svm = joblib.load("root/models/sa/svm_model.pkl")
    else: 
        svm = joblib.load("root/models/sa/final_svm_model.pkl")

    # Predict probabilities and sentiment
    svm_probs = svm.predict_proba(test_data)
    svm_sentiment = svm.predict(test_data)

    svm_probs_df = pd.DataFrame(data = svm_probs, columns = ['NEGATIVE', 'POSITIVE'])

    # Store SVM predictions into results dataframe
    results_svm = pd.DataFrame()
    results_svm['svm_sentiment'] = np.array(svm_sentiment)
    results_svm['svm_prob'] = svm_probs_df['POSITIVE']

    return results_svm



def XGB_predict(XGB_data):
    # Load the trained XGBoost model
    model_xgb =  XGBClassifier()
    model_xgb.load_model("models/sa/xgb_model.json")
    
    # Predict probabilities and sentiment
    xgb_probs = model_xgb.predict_proba(XGB_data)
    xgb_sentiment = model_xgb.predict(XGB_data)

    xgb_probs_df = pd.DataFrame(data = xgb_probs, columns = ['NEGATIVE', 'POSITIVE'])
    
    label_map_3 = {
    1 : 'positive',
    0 : 'negative',
    }

    # Store XGB predictions into results dataframe
    results_xgb = pd.DataFrame()
    results_xgb['xgb_sentiment'] = np.array(xgb_sentiment)
    # results_xgb['xgb_sentiment_class'] = results_xgb['xgb_sentiment'].map(label_map_3)
    results_xgb['xgb_prob'] = xgb_probs_df['POSITIVE']

    return results_xgb



def flair_predict(flair_data):
    # Load Flair model
    tagger = Classifier.load('sentiment')

    flair_prob = []
    flair_sentiments = []

    for review in flair_data['Text'].to_list():
    
        # Convert format of review to Sentence
        sentence = Sentence(review)
        
        # Make prediction using flair
        tagger.predict(sentence)
        
        # extract sentiment prediction
        flair_prob.append(sentence.labels[0].score)  # numerical score 0-1 (probability of class)
        flair_sentiments.append(sentence.labels[0].value)  # 'POSITIVE' or 'NEGATIVE' sentiment

    # Store the probability to predict positive class for each review
    flair_pos_probs = [0] *  len(flair_prob)

    for i in range(0,len(flair_prob)):
        if flair_sentiments[i] == "NEGATIVE":
            flair_pos_probs[i] = 1 - flair_prob[i]
        
        elif flair_sentiments[i] == "POSITIVE":
            flair_pos_probs[i] = flair_prob[i]

    label_map_1 = {
    'POSITIVE': 1,
    'NEGATIVE': 0,
    }

    
    # Create a new dataframe to store all results
    flair_results = pd.DataFrame()
    flair_results['flair_sentiment'] = np.array(flair_sentiments)
    flair_results['flair_sentiment'] = flair_results['flair_sentiment'].map(label_map_1)
    flair_results['flair_prob'] = np.array(flair_pos_probs)
    
    return flair_results



def ensemble_xgb_flair(SA_PROCESSED_DF_XGB, SA_PROCESSED_DF_FLAIR):
    '''
    inputs : DataFrames with processed data for XGBoost and Flair respectively
    output : DataFrame with final class predictions and probability of predictions
    '''

    time = SA_PROCESSED_DF_FLAIR['Time']
    text = SA_PROCESSED_DF_FLAIR['Text']
    ### Model 1: Flair
    flair_predictions = flair_predict(SA_PROCESSED_DF_FLAIR)

    ### Model 2: XGBoost
    XGB_predictions = XGB_predict(SA_PROCESSED_DF_XGB)

    # Create a new dataframe to store all results
    results = pd.DataFrame()
    results['flair_sentiment'] = flair_predictions['flair_sentiment']
    results['flair_prob'] = flair_predictions['flair_prob']

    results['xgb_sentiment'] = XGB_predictions['xgb_sentiment']
    results['xgb_prob'] = XGB_predictions['xgb_prob']
    
    label_map_3 = {
    1 : 'positive',
    0 : 'negative',
    }
    
    ## Final: Ensemble of Flair and XGBoost predictions
    results['avg_prob'] = (results['flair_prob'] + results['xgb_prob']) / 2
    results['final_sentiment'] = np.where(results['avg_prob'] > 0.5, 1, 0)
    results['Sentiment'] = results['final_sentiment'].map(label_map_3)

    results['Time'] = np.array(time)
    results['Text'] = np.array(text)
    return results # results['Sentiment'] is the final predicted sentiment (positive/negative)



def SA_MODEL_PREDICT(SA_PROCESSED_DF_SVM, SA_PROCESSED_DF_FLAIR, mode):
    '''
    inputs : DataFrames with processed data for SVMand Flair respectively, mode = train|predict
    output : DataFrame with final class predictions and probability of predictions
    '''
    time = SA_PROCESSED_DF_FLAIR['Time']
    text = SA_PROCESSED_DF_FLAIR['Text']
    ### Model 1: Flair
    flair_predictions = flair_predict(SA_PROCESSED_DF_FLAIR)

    ### Model 2: SVM
    svm_predictions = svm_predict(SA_PROCESSED_DF_SVM, mode)

    # Create a new dataframe to store all results
    results = pd.DataFrame()
    results['flair_sentiment'] = flair_predictions['flair_sentiment']
    results['flair_prob'] = flair_predictions['flair_prob']

    results['svm_sentiment'] = svm_predictions['svm_sentiment']
    results['svm_prob'] = svm_predictions['svm_prob']
    
    label_map_3 = {
    1 : 'positive',
    0 : 'negative',
    }
    
    ## Final: Ensemble of Flair and SVM predictions
    results['avg_prob'] = (results['flair_prob'] + results['svm_prob']) / 2
    results['final_sentiment'] = np.where(results['avg_prob'] > 0.5, 1, 0)
    results['Sentiment'] = results['final_sentiment'].map(label_map_3)

    results['Time'] = np.array(time)
    results['Text'] = np.array(text)
    return results # results['Sentiment'] is the final predicted sentiment (positive/negative)
