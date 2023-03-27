import requests
import os
import datetime
import pandas as pd
import fastapi
from utils import *
from io import StringIO
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import HTTPException

# For Model Deployment
description = """
Available endpoints are as listed below.  

### Notice
Not implemented yet
"""

# uvicorn main:app --port 5000, API Documentation: 127.0.0.1:5000/docs
app = fastapi.FastAPI(
    title = "Semantic Analysis and Topic Modelling",
    description = description,
    version = "0.0.1"
)

app.mount("/assets", StaticFiles(directory="assets"), name="assets")
templates = Jinja2Templates(directory="assets")

@app.get("/", response_class=HTMLResponse)
async def root(request: fastapi.Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
async def predict_upload_csv(file: fastapi.UploadFile = fastapi.File(...)):
    """
    Make predictions using the models for a updated csv file by the user

    Input: CSV File with the columns=["Time", "Text"], Zip File containing multiple CSV files of the previous formats
    
    Output: 3 downloadable CSV files 
    1 for Sentiment Analysis predictions
    2 for Topic Modelling predictions (Positive and Negative Sentiments)
    """
    CURRENT_TIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    expected_columns = set(["Time", "Text"])
    if file.content_type == "application/zip":
        master_df = zip_preprocess(file, expected_columns)
        if set(master_df.columns) != expected_columns:
            raise "Unexpected columns in DataFrame. Expected: {expected_columns}. Actual: {set(master_df.columns)}"
    elif file.content_type in ["text/csv", "application/csv"]:
        master_df = pd.read_csv(file.file)
        if set(master_df.columns) != expected_columns:
            raise "Unexpected columns in DataFrame. Expected: {expected_columns}. Actual: {set(master_df.columns)}"
    else:
        raise HTTPException(status_code=400, detail="Only Zip and CSV files are allowed")
    sa_pred, tm_pos_pred, tm_neg_pred = generate_predictions(master_df, CURRENT_TIME)
    sa_pred_file = StringIO(sa_pred)
    tm_pos_pred_file = StringIO(tm_pos_pred)
    tm_neg_pred_file = StringIO(tm_neg_pred)
    return {
        "sa_predictions.csv": sa_pred_file.get_value(),
        "tm_pos_predictions.csv": tm_pos_pred_file.get_value(),
        "tm_neg_predictions.csv": tm_neg_pred_file.get_value(),
        "submission_id": CURRENT_TIME
    }

@app.get("/api/raw_data")
async def get_raw_data(CURRENT_TIME: str):
    """
    Endpoint to look at the raw_data submitted in JSON format
    """
    if CURRENT_TIME:
        return Response(retrieve_raw_data(CURRENT_TIME).to_json(orient="records"), media_type="application/json")
    return None

@app.get("/api/data_report")
async def get_data_report(CURRENT_TIME: str):
    """
    Endpoint to look at the Basic EDA of the raw_data submitted
    """
    if CURRENT_TIME:
        return FileResponse(retrieve_data_report(CURRENT_TIME), media_type="text/html")
    return None

@app.get("/api/cleaned_data")
async def get_data_report(CURRENT_TIME: str):
    """
    Endpoint to look at the Basic EDA of the raw_data submitted
    """
    if CURRENT_TIME:
        return Response(retrieve_cleaned_data(CURRENT_TIME), media_type="text/html")
    return None

@app.get("/api/sa_pred")
async def get_semantic_predictions(CURRENT_TIME: str):
    """
    Endpoint to look at the predictions by the SA model 
    """
    if CURRENT_TIME:
        return Response(retrieve_sa_pred(CURRENT_TIME).to_json(orient="records"), media_type="application/json")
    return None

@app.get("/api/tm_pred")
async def get_tm_predictions(CURRENT_TIME: str, SENTIMENT="POS"):
    """
    Endpoint to look at the predictions by the TM model
    """
    if SENTIMENT not in ["POS", "NEG"]:
        raise "Unexpected Sentiment Value. Sentiment value must be 'POS' or 'NEG', DEFAULT: 'POS'"
    if CURRENT_TIME:
        return Response(retrieve_tm_pred(CURRENT_TIME, SENTIMENT.upper()).to_json(orient="records"), media_type="application/json")
    return None
