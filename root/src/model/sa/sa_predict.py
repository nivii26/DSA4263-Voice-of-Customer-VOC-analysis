# Basic requirements
import pandas as pd
import numpy as np

# For svm
from sklearn.svm import NuSVC 

# For flair
from flair.nn import Classifier
from flair.data import Sentence
import joblib

from ...preprocessing.sa_preprocessing import SA_PREPROCESS_TEST


def svm_predict(test_data_svm, mode):
    '''
    input : processed test_data DataFrame for SVM model
    output : results of SVM model 
    function : use SVM model to predict the sentiment results
    '''

    # Load the trained SVM model
    if mode == 'train':
        svm = joblib.load("root/models/sa/svm_model.pkl")
    else: 
        svm = joblib.load("root/models/sa/final_svm_model.pkl")

    # Predict probabilities and sentiment
    svm_probs = svm.predict_proba(test_data_svm)
    svm_sentiment = svm.predict(test_data_svm)

    svm_probs_df = pd.DataFrame(data = svm_probs, columns = ['NEGATIVE', 'POSITIVE'])

    # Store SVM predictions into results dataframe
    svm_results = pd.DataFrame()
    svm_results['svm_sentiment'] = np.array(svm_sentiment)
    svm_results['svm_prob'] = svm_probs_df['POSITIVE']

    return svm_results



def flair_predict(test_data_flair):
    '''
    input : processed test_data DataFrame for flair model
    output : results of flair model 
    function : use flair model to predict the sentiment results
    '''

    # Load Flair model
    tagger = Classifier.load('sentiment')

    flair_prob = []
    flair_sentiments = []

    for review in test_data_flair['Text'].to_list():
    
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



def SA_MODEL_PREDICT(SA_PROCESSED_DF_SVM, SA_PROCESSED_DF_FLAIR, mode):
    '''
    inputs : DataFrames with processed data for SVM and Flair respectively, mode = train|predict
    output : DataFrame with final class predictions and probability of predictions
    function: ensemble the results of two best performance models and give the predictions
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

def scoring(test_df):
    '''
    inputs : DataFrame with 'Time' and 'Text'
    output :  dataframe["Text", Time", "predicted_sentiment_probability", "predicted_sentiment"]
    function: Apply preprocessing and fit final model to output final_sentiment and predicted sentiment_probability
    '''
    # SA Preprocessing
    SA_PROCESSED_DF_SVM, SA_PROCESSED_DF_FLAIR = SA_PREPROCESS_TEST(test_df)

    # SA Predictions
    SA_PREDICTIONS_DF = SA_MODEL_PREDICT(SA_PROCESSED_DF_SVM, SA_PROCESSED_DF_FLAIR, "predict")
    SA_PREDICTIONS_DF = SA_PREDICTIONS_DF[["Time", "Text", "avg_prob", "Sentiment"]]
  
    # Rename columns to desired outputs
    SA_PREDICTIONS_DF.rename(columns={"avg_prob":"predicted_sentiment_probability", "Sentiment":"predicted_sentiment"}, inplace=True)

    # Save/return results
    SA_PREDICTIONS_DF.to_csv("reviews_test_predictions_CAJN.csv", index = False)

    return SA_PREDICTIONS_DF
