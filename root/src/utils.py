import os
import pandas as pd
import requests
import zipfile

from preprocessing import TM_PREPROCESS_TEST, SA_PREPROCESS_TEST
from model import SA_MODEL_PREDICT
from model import TM_MODEL_PREDICT

def zip_preprocess(zip_file, expected_columns):
    """
    Unzip zipfile, extract and combine all relevant csv file into one DataFrame

    Input: zipfile

    Output: 1 DataFrame with columns ["Date", "Text"] created using all relevant csv files in the zipfile
    """
    masterdf = pd.DataFrame(columns=list(expected_columns))
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith("csv"):
                with zip_ref.open(file) as f:
                    tmp_df = pd.read_csv(f)
                    if set(tmp_df.columns) != set(expected_columns):
                        continue
                    f.close()
                    masterdf = pd.concat([masterdf, tmp_df])
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
    RAW -> ./root/data/raw
    CLEANED -> ./root/data/processed
    SA -> ./root/src/data/sa
    TM -> ./root/src/data/tm

    Output:
    1. SA_PREDICTIONS_DF: DataFrame ["Time", "Text", "Sentiment", "avg_prob"] # Sentiment Values: "positive" or "negative"
    2. TM_PREDICTIONS_DF: DataFrame for positive sentiments ["Time", "Text", "Sentiment", "Predicted Topic"]
    """
    # SA Preprocessing
    SA_PROCESSED_DF_SVM, SA_PROCESSED_DF_FLAIR = SA_PREPROCESS_TEST(RAW_DF)
    # SA Predictions
    SA_PREDICTIONS_DF = SA_MODEL_PREDICT(SA_PROCESSED_DF_SVM, SA_PROCESSED_DF_FLAIR, "predict")
    # TM Preprocessing
    TM_DF = TM_PREPROCESS_TEST(SA_PREDICTIONS_DF)
    # TM Predictions
    TM_PREDICTIONS_DF = TM_MODEL_PREDICT(TM_DF)
    if SAVE:
        RAW_DF.to_csv(fr"./root/data/raw/{CURRENT_TIME}_RAW_DF.csv", index=False)
        SA_PROCESSED_DF_SVM.to_csv(fr"./root/src/data/sa/{CURRENT_TIME}_SA_PROCESSED_DF_SVM.csv", index=False)
        SA_PROCESSED_DF_FLAIR.to_csv(fr"./root/src/data/sa/{CURRENT_TIME}_SA_PROCESSED_DF_FLAIR.csv", index=False)
        SA_PREDICTIONS_DF.to_csv(fr"./root/src/data/sa/{CURRENT_TIME}_SA_PRED_DF.csv", index=False)
        TM_DF.to_csv(fr"./root/src/data/tm/{CURRENT_TIME}_TM_DF.csv", index=False)
        TM_PREDICTIONS_DF.to_csv(fr"./root/src/data/tm/{CURRENT_TIME}_TM_PRED_DF.csv", index=False)
    return SA_PREDICTIONS_DF, TM_PREDICTIONS_DF

# Retrieving results
def retrieve_raw_data(CURRENT_TIME):
    """
    Support function to retreieve the raw data from local data folder when raw data endpoint is called
    """ 
    if f"{CURRENT_TIME}_RAW_DF.csv" in os.listdir(r"./root/data/raw"):
        data = pd.read_csv(fr"./root/data/raw/{CURRENT_TIME}_RAW_DF.csv")
        return data
    return None

def retrieve_sa_pred(CURRENT_TIME):
    """
    Support function to retreieve the SA prediction data from local data folder when raw data endpoint is called
    """ 
    if f"{CURRENT_TIME}_SA_PRED_DF.csv" in os.listdir(r"./root/src/data/sa"):
        data = pd.read_csv(fr"./root/src/data/sa/{CURRENT_TIME}_SA_PRED_DF.csv")
        return data
    return None

def retrieve_tm_pred(CURRENT_TIME):
    """
    Support function to retreieve the TM prediction data from local data folder when raw data endpoint is called
    """ 
    if f"{CURRENT_TIME}_TM_PRED_DF.csv" in os.listdir(r"./root/src/data/tm"):
        data = pd.read_csv(fr"./root/src/data/tm/{CURRENT_TIME}_TM_PRED_DF.csv")
        return data
    return None

# def retrieve_cleaned_data(CURRENT_TIME):
#     if f"{CURRENT_TIME}_CLEANED_DF.csv" in os.listdir(r"./root/data/processed"):
#         data = pd.read_csv(fr"./root/data/processed/{CURRENT_TIME}_CLEANED_DF.csv")
#         return data
#     return None

# def retrieve_data_report(CURRENT_TIME):
#     if f"{CURRENT_TIME}_DATA_REPORT.html" in os.listdir(r"./root/data/processed/report"):
#         path = fr"./root/data/processed/report/{CURRENT_TIME}_DATA_REPORT.html"
#         return path
#     return None

# Access Endpoints
def predict_file(url, dir, fname):
    """
    Function to make a post request to the predict endpoint

    Input: 
    1. url of the website homepage
    2. dir: directory of the file you are predicting
    3. fname: filename of the file 

    Output:
    1. Zip file containing the SA and TM prediction saved in your directory
    """
    endpoint = "predict"
    url = f"{url}/{endpoint}"
    file = open(f"{dir}/{fname}", "rb")

    response = requests.post(url, files={"file":(fname,file)})

    if response.status_code == 200:
        with open("test_predictions.zip","wb") as pred_file:
            pred_file.write(response.content)
            pred_file.close()
    else:
        print("Error!", response.text)
    file.close()

def request_raw_data(url, id):
    """
    Function to make a get request to the raw_data endpoint

    Input: 
    1. url of the website homepage
    2. id: CURRENT_TIME when the predict function was called

    Output:
    1. JSON formatted content of the raw data file
    """
    endpoint = "raw_data"
    url = f"{url}/{endpoint}"
    response  = requests.get(url, params={"CURRENT_TIME":{id}})
    if response.status_code == 200:
        return response.content
    else:
        print("Error!", response.text)
    
def request_sa_pred_data(url, id):
    """
    Function to make a get request to the SA_PRED endpoint

    Input: 
    1. url of the website homepage
    2. id: CURRENT_TIME when the predict function was called

    Output:
    1. JSON formatted content of the SA_PRED data file
    """
    endpoint = "sa_pred"
    url = f"{url}/{endpoint}"
    response  = requests.get(url, params={"CURRENT_TIME":{id}})
    if response.status_code == 200:
        return response.content
    else:
        print("Error!", response.text)

def request_tm_pred_data(url, id):
    """
    Function to make a get request to the TM_PRED endpoint

    Input: 
    1. url of the website homepage
    2. id: CURRENT_TIME when the predict function was called

    Output:
    1. JSON formatted content of the TM_PRED data file
    """
    endpoint = "tm_pred"
    url = f"{url}/{endpoint}"
    response  = requests.get(url, params={"CURRENT_TIME":{id}})
    if response.status_code == 200:
        return response.content
    else:
        print("Error!", response.text)

# def request_data_report(url, id):
#     endpoint = "data_report"
#     url = f"{url}/{endpoint}"
#     response  = requests.get(url, params={"CURRENT_TIME":{id}})
#     if response.status_code == 200:
#         return response.content
#     else:
#         print("Error!", response.text)

# def request_cleaned_data(url, id):
#     endpoint = "cleaned_data"
#     url = f"{url}/{endpoint}"
#     response  = requests.get(url, params={"CURRENT_TIME":{id}})
#     if response.status_code == 200:
#         return response.content
#     else:
#         print("Error!", response.text)


