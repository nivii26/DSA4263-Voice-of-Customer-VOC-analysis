import datetime
import pandas as pd
import fastapi
import zipfile
from utils import *
from io import BytesIO
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import HTTPException

# For Model Deployment
description = """
Available endpoints are as listed below.  

### Notice
Not implemented yet
"""

# uvicorn --app-dir=./root/src main:app --reload --port 5000, API Documentation: 127.0.0.1:5000/docs
app = fastapi.FastAPI(
    title = "Semantic Analysis and Topic Modelling",
    description = description,
    version = "0.0.1"
)

app.mount("/assets", StaticFiles(directory="./root/src/assets"), name="assets")
templates = Jinja2Templates(directory='./root/src/assets')

@app.get("/", response_class=HTMLResponse)
async def root(request: fastapi.Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
async def predict_upload_csv(file: fastapi.UploadFile = fastapi.File(...)):
    """
    Make predictions using the models for a updated csv file by the user

    Input: CSV File with the columns=["Time", "Text"], Zip File containing multiple CSV files of the previous formats
    
    Output: 2 downloadable CSV files 
    1 for Sentiment Analysis predictions
    2 for Topic Modelling predictions
    """
    CURRENT_TIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    expected_columns = set(["Time", "Text"])
    if file.filename.endswith(".zip"):
    #if file.content_type in ("application/zip", "application/x-zip-compressed"):
        content = file.file.read()
        fileReadBuffer = BytesIO(content)
        master_df = zip_preprocess(fileReadBuffer, expected_columns)
        if set(master_df.columns) != expected_columns:
            raise f"Unexpected columns in DataFrame. Expected: {expected_columns}. Actual: {set(master_df.columns)}"
    elif file.filename.endswith(".csv"):
    #elif file.content_type in ["text/csv", "application/csv"]:
        master_df = pd.read_csv(file.file)
        if set(master_df.columns) != expected_columns:
            raise f"Unexpected columns in DataFrame. Expected: {expected_columns}. Actual: {set(master_df.columns)}"
    else:
        raise HTTPException(status_code=400, detail="Only Zip and CSV files are allowed")
    sa_pred, tm_pred = generate_predictions(master_df, CURRENT_TIME)
    sa_result_pred = sa_pred.loc[:,["Time", "Text", "avg_prob", "Sentiment"]]
    sa_result_pred.rename(columns={"avg_prob":"predicted_sentiment_probability", "Sentiment":"predicted_sentiment"})
    tm_result_pred = tm_pred.loc[:,["Time", "Text", "Sentiment", "Predicted Topic"]]
    dfs = [("SA_RESULT.csv", sa_result_pred), ("TM_RESULT.csv", tm_result_pred)]
    inMemoryBuffer = BytesIO()
    with zipfile.ZipFile(inMemoryBuffer, mode="w") as zip_file:
        for file_name, df in dfs:
            csv_bytes = df.to_csv(index=False).encode()
            zip_file.writestr(file_name, csv_bytes)
    response = Response(content=inMemoryBuffer.getvalue(),
                        media_type="application/zip")
    response.headers["Content-Disposition"] = f"attachment; filename={CURRENT_TIME}_predictions.zip"
    return response

@app.get("/api/raw_data")
async def get_raw_data(CURRENT_TIME: int):
    """
    Endpoint to look at the raw_data submitted in JSON format
    """
    if CURRENT_TIME:
        return Response(retrieve_raw_data(CURRENT_TIME).to_json(orient="records"), media_type="application/json")
    return None

# @app.get("/api/data_report")
# async def get_data_report(CURRENT_TIME: int):
#     """
#     Endpoint to look at the Basic EDA of the raw_data submitted
#     """
#     if CURRENT_TIME:
#         return FileResponse(retrieve_data_report(CURRENT_TIME), media_type="text/html")
#     return None

# @app.get("/api/cleaned_data")
# async def get_cleaned_data(CURRENT_TIME: int):
#     """
#     Endpoint to look at the Basic EDA of the raw_data submitted
#     """
#     if CURRENT_TIME:
#         return Response(retrieve_cleaned_data(CURRENT_TIME).to_json(orient="records"), media_type="application/json")
#     return None

@app.get("/api/sa_pred")
async def get_semantic_predictions(CURRENT_TIME: int):
    """
    Endpoint to look at the predictions by the SA model 
    """
    if CURRENT_TIME:
        return Response(retrieve_sa_pred(CURRENT_TIME).to_json(orient="records"), media_type="application/json")
    return None

@app.get("/api/tm_pred")
async def get_tm_predictions(CURRENT_TIME: int):
    """
    Endpoint to look at the predictions by the TM model
    """
    if CURRENT_TIME:
        return Response(retrieve_tm_pred(CURRENT_TIME).to_json(orient="records"), media_type="application/json")
    return None