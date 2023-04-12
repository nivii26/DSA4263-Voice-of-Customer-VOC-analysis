import pytest
from ..src.model.sa.sa_predict import *
import pandas as pd
from ..src.preprocessing.rawdata_preprocessing import *
from ..src.preprocessing.sa_preprocessing import *


@pytest.fixture
def sample_documents():
    sentiment = ["positive", "negative"]
    time = ["18/6/21", "29/7/19"]
    text = [
        "This is a very healthy dog food. Good for their digestion.",
        "THis product is definitely not as good as some other gluten free cookies!",
    ]
    df = pd.DataFrame({"Time": time, "Text": text})
    df_xgb, df_flair = SA_PREPROCESS_TEST(df)
    return df_xgb, df_flair


def test_svm(sample_documents):
    df_xgb, df_flair = sample_documents
    prediction = svm_predict(df_xgb,"test")

    assert isinstance(prediction, pd.DataFrame)

    flag_type = False
    for row in prediction.iterrows():
        if isinstance(row[1], pd.Series):
            if isinstance(row[1].values[0], float) and isinstance(row[1].values[1], float):
                flag_type = True
    assert flag_type is True

    flag_label = False
    for row in prediction.iterrows():
        if int(row[1].values[0])==0 or int(row[1].values[0])==1:
            if row[1].values[1]<=1 and row[1].values[1]>=0:
                flag_label = True
    assert flag_label is True


def test_xgb(sample_documents):
    df_xgb, df_flair = sample_documents
    prediction = XGB_predict(df_xgb)

    assert isinstance(prediction, pd.DataFrame)
    flag_type = False
    for row in prediction.iterrows():
        if isinstance(row[1], pd.Series):
            if isinstance(row[1].values[0], float) and isinstance(row[1].values[1], float):
                flag_type = True
    assert flag_type is True

    flag_label = False
    for row in prediction.iterrows():
        if int(row[1].values[0])==0 or int(row[1].values[0])==1:
            if row[1].values[1]<=1 and row[1].values[1]>=0:
                flag_label = True
    assert flag_label is True

def test_flair(sample_documents):
    df_xgb, df_flair = sample_documents
    prediction = flair_predict(df_flair)

    assert isinstance(prediction, pd.DataFrame)
    flag_type = False
    for row in prediction.iterrows():
        if isinstance(row[1], pd.Series):
            if isinstance(row[1].values[0], float) and isinstance(row[1].values[1], float):
                flag_type = True
    assert flag_type is True

    flag_label = False
    for row in prediction.iterrows():
        if int(row[1].values[0])==0 or int(row[1].values[0])==1:
            if row[1].values[1]<=1 and row[1].values[1]>=0:
                flag_label = True
    assert flag_label is True

def test_merge(sample_documents):
    df_xgb, df_flair = sample_documents
    prediction = ensemble_xgb_flair(df_xgb, df_flair)

    assert isinstance(prediction, pd.DataFrame)
    flag_type = False
    for row in prediction.iterrows():
        if isinstance(row[1], pd.Series):
            if isinstance(row[1].values[0], int) and isinstance(row[1].values[1], float) and isinstance(row[1].values[2], int) and isinstance(row[1].values[3], float) and isinstance(row[1].values[4], float) and isinstance(row[1].values[5], int) and isinstance(row[1].values[6], str) and isinstance(row[1].values[7], str) and isinstance(row[1].values[8], str):
                flag_type = True
    assert flag_type is True

    flag_label = True
    for row in prediction.iterrows():
        for i in (0,2,5):
            if int(row[1].values[i])!=0 and int(row[1].values[i])!=1:
                flag_label = False
        for i in (1,3,4):
            if row[1].values[i]>1 or row[1].values[i]<0:
                flag_label = False
        if row[1].values[6]!="positive" and row[1].values[6]!= "negative":
                flag_label = False
    assert flag_label is True





def test_final(sample_documents):
    df_xgb, df_flair = sample_documents
    prediction = SA_MODEL_PREDICT(df_xgb, df_flair, "test")

    assert isinstance(prediction, pd.DataFrame)
    flag_type = False
    for row in prediction.iterrows():
        if isinstance(row[1], pd.Series):
            if isinstance(row[1].values[0], int) and isinstance(row[1].values[1], float) and isinstance(row[1].values[2], int) and isinstance(row[1].values[3], float) and isinstance(row[1].values[4], float) and isinstance(row[1].values[5], int) and isinstance(row[1].values[6], str) and isinstance(row[1].values[7], str) and isinstance(row[1].values[8], str):
                flag_type = True
    assert flag_type is True

    flag_label = True
    for row in prediction.iterrows():
        for i in (0,2,5):
            if int(row[1].values[i])!=0 and int(row[1].values[i])!=1:
                flag_label = False
        for i in (1,3,4):
            if row[1].values[i]>1 or row[1].values[i]<0:
                flag_label = False
        if row[1].values[6]!="positive" and row[1].values[6]!= "negative":
                flag_label = False
    assert flag_label is True



# if __name__ == "__main__":
#     # os.chdir(r"./root/unit_testing")
#     df_xgb, df_flair = sample_documents()
#     test_svm(df_xgb)
#     test_xgb(df_xgb)
#     test_flair(df_flair)
#     test_merge(df_xgb, df_flair)
#     test_final(df_xgb, df_flair)
#     print("All Tests Passed")
