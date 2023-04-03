# Basic requirements
import pandas as pd
import numpy as np

# For XGBoost
from xgboost import XGBClassifier
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

# For flair
from flair.nn import Classifier
from flair.data import Sentence


def sa_model_predict(SA_PROCESSED_DF_XGB, SA_PROCESSED_DF_FLAIR):
    '''
    inputs : DataFrames with processed data for XGBoost and Flair respectively
    output : DataFrame with final class predictions and probability of predictions
    '''

    ### Model 1: Flair

    # Load Flair model
    tagger = Classifier.load('sentiment')

    flair_prob = []
    flair_sentiments = []

    for review in  SA_PROCESSED_DF_FLAIR['Text'].to_list():
    
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

    label_map_2 = {
    'POSITIVE': 1,
    'NEGATIVE': 0,
    }
    
    # Create a new dataframe to store all results
    results = pd.DataFrame()
    results['flair_sentiment'] = np.array(flair_sentiments)
    results['flair_sentiment'] = results['flair_sentiment'].map(label_map_2)
    results['flair_prob'] = np.array(flair_pos_probs)
    
    ### Model 2: XGBoost

    # Load the trained XGBoost model
    model_xgb = XGBClassifier()
    model_xgb.load_model("xgb_model.json")

    # Predict probabilities and sentiment
    xgb_probs = model_xgb.predict_proba(SA_PROCESSED_DF_XGB)
    xgb_sentiment = model_xgb.predict(SA_PROCESSED_DF_XGB)

    xgb_probs_df = pd.DataFrame(data = xgb_probs, columns = ['NEGATIVE', 'POSITIVE'])

    # Store XGB predictions into results dataframe
    results['xgb_sentiment'] = np.array(xgb_sentiment)
    results['xgb_prob'] = xgb_probs_df['POSITIVE']

    ## Final: Ensemble of Flair and XGBoost predictions
    results['avg_prob'] = (results['flair_prob'] + results['xgb_prob']) / 2
    results['final_sentiment'] = np.where(results['avg_prob'] > 0.5, 1, 0)
    
    return results
