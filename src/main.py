import requests
import os
import pandas as pd
import fastapi
from utils import *
from io import StringIO
from fastapi.responses import HTMLResponse, ORJSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# For Model Deployment
description = """
Available endpoints are as listed below.  

### Notice
Not implemented yet
"""

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

    Input: CSV File with the columns=["Sentiment", "Time", "Text"]
    
    Output: 2 downloadable CSV files (1 for Sentiment Analysis predictions, 1 for Topic Modelling predictions)
    """
    master_df = pd.read_csv(file.file)
    sa_pred, tm_pred = generate_predictions(master_df)
    sa_pred_file = StringIO(sa_pred)
    tm_pred_file = StringIO(tm_pred)
    return {
        "sa_predictions.csv": sa_pred_file.get_value(),
        "tm_predictions.csv": tm_pred_file.get_value() 
    }

@app.get("/api/data")
async def get_data():
    """
    Endpoint to look at the raw_data submitted in JSON format
    """
    return Response(retrieve_data().to_json(orient="records"), media_type="application/json")

@app.get("/api/data_report")
async def get_data_report():
    """
    Endpoint to look at the Basic EDA of the raw_data submitted
    """
    return Response(retrieve_data_report(), media_type="text/html")

@app.get("/api/sa_pred")
async def get_semantic_predictions():
    """
    Endpoint to look at the predictions by the SA model 
    """
    return Response(retrieve_sa_pred().to_json(orient="records"), media_type="application/json")

@app.get("/api/tm_pred")
async def get_tm_predictions():
    """
    Endpoint to look at the predictions by the TM model 
    """
    return Response(retrieve_tm_pred().to_json(orient="records"), media_type="application/json")

