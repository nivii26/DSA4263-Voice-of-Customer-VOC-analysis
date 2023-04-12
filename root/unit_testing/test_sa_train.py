import pytest
from ..src.model.sa.sa_train import *
import os
import math
import pandas as pd
from ..src.preprocessing.rawdata_preprocessing import *
from ..src.preprocessing.sa_preprocessing import *


def sample_documents(df):
    df_train, df_test = sa_train_test_split(df)
    df_train = SA_PREPROCESS_TRAIN(df_train)
    df_test_xgb, df_test_flair = SA_PREPROCESS_TEST(df_test)
    return df_train, df_test_xgb, df_test_flair


def test_sa_train_test_split(input_df):
    train_data, test_data = sa_train_test_split(input_df)

    assert isinstance(train_data, pd.DataFrame)
    assert isinstance(test_data, pd.DataFrame)

    flag_ratio = False
    if math.isclose(len(train_data.index)/len(test_data.index), 7/3, rel_tol=1, abs_tol=0.0):
        flag_ratio = True
    assert flag_ratio is True


def test_train_XGB(df_train, df_test):
    print("Test for train_XGB.")
    train_XGB(df_train, df_test)
    model_path = 'root/models/sa/xgb_model.json'
    flag = False
    if os.path.exists(model_path):
        flag = True
    assert flag is True

def test_final_svm_full_model(df_train):
    print("Test for final_svm_full_model.")
    final_svm_full_model(df_train)

    model_path = 'root/models/sa/final_svm_model.pkl'
    flag = False
    if os.path.exists(model_path):
        flag = True
    assert flag is True
        


if __name__ == "__main__":
    #os.chdir(r"./root/unit_testing")
    input_df = pd.read_csv(r"root/unit_testing/testcase/input/test_sa_input.csv")
    df_train, df_test_xgb, df_test_flair = sample_documents(input_df)

    test_sa_train_test_split(input_df)
    test_train_XGB(df_train, df_test_xgb)
    test_final_svm_full_model(df_train)

    print("All Tests Passed")