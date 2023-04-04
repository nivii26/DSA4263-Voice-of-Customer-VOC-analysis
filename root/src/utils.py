import os
import pandas as pd
import zipfile

from .preprocessing import TM_PREPROCESS_TEST, SA_PREPROCESS_TEST
from .model.sa import SA_MODEL_PREDICT
from .model.tm import TM_MODEL_PREDICT

def zip_preprocess(zip_file, expected_columns):
    """
    Unzip zipfile, extract and combine all relevant csv file into one DataFrame

    Input: zipfile

    Output: 1 DataFrame with columns ["Date", "Text"] created using all relevant csv files in the zipfile
    """
    masterdf = pd.DataFrame(columns=["Time", "Text"])
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file in zip_ref.infolist():
            if file.filename.endswith("csv"):
                with zip_ref.open(file) as f:
                    tmp_df = pd.read_csv(f)
                    if set(tmp_df.columns) != set(expected_columns):
                        continue
                    f.close()
                    master_df = pd.concat([master_df, tmp_df])
    return masterdf

def generate_predictions(RAW_DF, CURRENT_TIME, SAVE=True):
    """
    Generate a Sentiment and Topic Modelling predictions from raw data
    Pipeline:
    RawData -> CleanedData -> SA_Preprocessing -> SA_Predictions_df -> TM_Preprocessing -> TM_Predictions_df 
    -> Output (SA_Predictions_df, TM_Predictions_df [Pos, Neg])

    Input: 
    1. Raw DataFrame with columns ["Time", "Text"]
    2. CURRENT_TIME at which the request is made
    3. If SAVE=True, save a copy of all DF name tracked using CURRENT_TIME
    RAW -> ../data/raw
    CLEANED -> ../data/processed
    SA -> ./data/sa
    TM -> ./data/tm

    Output:
    1. SA_PREDICTIONS_DF: DataFrame ["Time", "Text", "Sentiment"] # Sentiment Values: "positive" or "negative"
    2. TM_POS_PRED_DF: DataFrame for positive sentiments ["Time", "Text", "Sentiment", "Topic"]
    3. TM_NEG_PRED_DF: DataFrame for negative sentiments ["Time", "Text", "Sentiment", "Topic"]
    """
    # SA Preprocessing
    SA_PROCESSED_DF_XGB, SA_PROCESSED_DF_FLAIR = SA_PREPROCESS_TEST(RAW_DF)
    # SA Predictions
    SA_PREDICTIONS_DF = SA_MODEL_PREDICT(SA_PROCESSED_DF_XGB, SA_PROCESSED_DF_FLAIR)
    # TM Preprocessing
    TM_POS_DF, TM_NEG_DF = TM_PREPROCESS_TEST(SA_PREDICTIONS_DF) 
    # TM Predictions
    TM_POS_PRED_DF = TM_MODEL_PREDICT(TM_POS_DF, "Positive")
    TM_NEG_PRED_DF = TM_MODEL_PREDICT(TM_NEG_DF, "Negative")
    if SAVE:
        RAW_DF.to_csv(fr"../data/raw/{CURRENT_TIME}_RAW_DF.csv", index=False)
        SA_PROCESSED_DF_XGB.to_csv(fr"./data/sa/{CURRENT_TIME}_SA_PROCESSED_DF_XGB.csv", index=False)
        SA_PROCESSED_DF_FLAIR.to_csv(fr"./data/sa/{CURRENT_TIME}_SA_PROCESSED_DF_FLAIR.csv", index=False)
        SA_PREDICTIONS_DF.to_csv(fr"./data/sa/{CURRENT_TIME}_SA_PRED_DF.csv", index=False)
        TM_POS_DF.to_csv(fr"./data/tm/{CURRENT_TIME}_TM_POS_DF.csv", index=False)
        TM_NEG_DF.to_csv(fr"./data/tm/{CURRENT_TIME}_TM_NEG_DF.csv", index=False)
        TM_POS_PRED_DF.to_csv(fr"./data/tm/{CURRENT_TIME}_TM_POS_PRED_DF.csv", index=False)
        TM_NEG_PRED_DF.to_csv(fr"./data/tm/{CURRENT_TIME}_TM_NEG_PRED_DF.csv", index=False)
    return SA_PREDICTIONS_DF, TM_POS_PRED_DF, TM_NEG_PRED_DF

# Retrieving results
def retrieve_raw_data(CURRENT_TIME):
    if f"{CURRENT_TIME}_RAW_DF.csv" in os.listdir(r"../data/raw"):
        data = pd.read_csv(fr"../data/raw/{CURRENT_TIME}_RAW_DF.csv")
        return data
    return None

def retrieve_cleaned_data(CURRENT_TIME):
    if f"{CURRENT_TIME}_CLEANED_DF.csv" in os.listdir(r"../data/processed"):
        data = pd.read_csv(fr"../data/processed/{CURRENT_TIME}_CLEANED_DF.csv")
        return data
    return None

def retrieve_data_report(CURRENT_TIME):
    if f"{CURRENT_TIME}_DATA_REPORT.html" in os.listdir(r"../data/processed/report"):
        path = fr"../data/processed/report/{CURRENT_TIME}_DATA_REPORT.html"
        return path
    return None

def retrieve_sa_pred(CURRENT_TIME):
    if f"{CURRENT_TIME}_SA_PRED_DF.csv" in os.listdir(r"./data/sa"):
        data = pd.read_csv(fr"./data/sa/{CURRENT_TIME}_SA_PRED_DF.csv")
        return data
    return None

def retrieve_tm_pred(CURRENT_TIME, SENTIMENT):
    if f"{CURRENT_TIME}_TM_{SENTIMENT}_PRED_DF.csv" in os.listdir(r"./data/tm"):
        data = pd.read_csv(fr"./data/tm/{CURRENT_TIME}_TM_{SENTIMENT}_PRED_DF.csv")
        return data
    return None

# TODO: Implement Visualizations 

# TODO: Test the API TO SEE IF IT WORKS -> NEED MODEL TO BE CREATED AND LOADED in MODEL_PREDICT


