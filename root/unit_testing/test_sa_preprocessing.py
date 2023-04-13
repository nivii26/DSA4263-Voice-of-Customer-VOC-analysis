import pytest
import math
import os
import pandas as pd
from ..src.preprocessing.rawdata_preprocessing import *
from ..src.preprocessing.sa_preprocessing import *


def test_augment_train():
    # create a sample dataframe
    train_data = pd.DataFrame({
        'Text': ['I love this movie', 'This movie is terrible'],
        'Sentiment': ['positive', 'negative']
    })
    # augment the data
    augmented_data = augment_train(train_data)
    # check if the augmented dataframe has more rows than the original
    assert len(augmented_data) > len(train_data)
    # check if the augmented dataframe has the same number of positive examples
    assert len(augmented_data[augmented_data['Sentiment'] == 'positive']) == len(train_data[train_data['Sentiment'] == 'positive'])
    # check if the augmented dataframe has more negative examples
    assert len(augmented_data[augmented_data['Sentiment'] == 'negative']) > len(train_data[train_data['Sentiment'] == 'negative'])

def test_sa_preprocess():
    text = "Here are some words for sample."
    expected_tokens = ['Here',  'word', 'sample', '.']
    ## remove stop words: are, some, for
    ## lemma: words->word
    assert sa_preprocess(text) == expected_tokens

@pytest.fixture
def sample_document():
    input_df = pd.read_csv(r"./root/unit_testing/testcase/input/test_sa_input.csv")
    input_df2 = pd.read_csv(r"./root/unit_testing/testcase/input/test_sa_input2.csv")
    return input_df, input_df2

def test_PREPROCESS_XGB(sample_document):
    for input_df in sample_document:
        features_df = PREPROCESS_XGB(input_df)

        # Test column names of features_df, except for 'Sentiment' column, should contain "PC" string
        expected_cols = [col for col in features_df.columns if 'PC' in col and col != 'Sentiment']
        cols = [col for col in features_df.columns if col != 'Sentiment']
        assert all(col in expected_cols for col in cols), \
            f"Column names do not meet expectations. Expected column names containing 'PC': {expected_cols}, Actual column names: {list(features_df.columns)}"

        # Test if 'Sentiment' column, if present, contains only 0 or 1 values
        if 'Sentiment' in features_df.columns:
            sentiment_values = features_df['Sentiment'].unique()
            assert all(val in [0, 1] for val in sentiment_values), \
                f"Values in 'Sentiment' column do not meet expectations. Expected values: [0, 1], Actual values: {list(sentiment_values)}"

        # Test the number of rows in features_df should be the same as input_df
        expected_row_count = len(input_df)
        assert len(features_df) == expected_row_count, \
            f"Number of rows does not meet expectations. Expected row count: {expected_row_count}, Actual row co0unt: {len(features_df)}"



def test_PREPROCESS_FLAIR(sample_document):
    for input_df in sample_document:
        cleaned_df = PREPROCESS_FLAIR(input_df)

        # Test if 'Time' and 'Text' columns are present in cleaned_df 
        expected_cols = ['Time', 'Text']
        assert all(col in cleaned_df.columns for col in expected_cols), \
            f"Expected columns: {expected_cols}, Actual columns: {list(cleaned_df.columns)}"
        
        # Test if 'Sentiment' column, if present, contains only 0 or 1 values
        if 'Sentiment' in cleaned_df.columns:
            sentiment_values = cleaned_df['Sentiment'].unique()
            assert all(val in [0, 1] for val in sentiment_values), \
                f"Values in 'Sentiment' column do not meet expectations. Expected values: [0, 1], Actual values: {list(sentiment_values)}"

        # Test if the number of rows in cleaned_df is the same as input_df
        expected_row_count = len(input_df)
        assert len(cleaned_df) == expected_row_count, \
            f"Number of rows does not meet expectations. Expected row count: {expected_row_count}, Actual row count: {len(cleaned_df)}"


def test_SA_PREPROCESS_TRAIN(sample_document):
    for input_df in sample_document:
        features_df = SA_PREPROCESS_TRAIN(input_df)

        # Test column names of features_df, except for 'Sentiment' column, should contain "PC" string
        expected_cols = [col for col in features_df.columns if 'PC' in col and col != 'Sentiment']
        cols = [col for col in features_df.columns if col != 'Sentiment']
        assert all(col in expected_cols for col in cols), \
            f"Column names do not meet expectations. Expected column names containing 'PC': {expected_cols}, Actual column names: {list(features_df.columns)}"

        # Test values in 'Sentiment' column should be either 0 or 1
        sentiment_values = features_df['Sentiment'].unique()
        assert all(val in [0, 1] for val in sentiment_values), \
            f"Values in 'Sentiment' column do not meet expectations. Expected values: [0, 1], Actual values: {list(sentiment_values)}"

def test_SA_PREPROCESS_TEST(sample_document):
    for input_df in sample_document:
        SA_PROCESSED_DF_XGB, SA_PROCESSED_DF_FLAIR = SA_PREPROCESS_TEST(input_df)
        assert isinstance(SA_PROCESSED_DF_XGB, pd.DataFrame)
        assert isinstance(SA_PROCESSED_DF_FLAIR, pd.DataFrame)

def test_final_full_data_preprocess(sample_document):
    for input_df in sample_document:
        features_df = final_full_data_preprocess(input_df)

        # Test column names of features_df, except for 'Sentiment' column, should contain "PC" string
        expected_cols = [col for col in features_df.columns if 'PC' in col and col != 'Sentiment']
        cols = [col for col in features_df.columns if col != 'Sentiment']
        assert all(col in expected_cols for col in cols), \
            f"Column names do not meet expectations. Expected column names containing 'PC': {expected_cols}, Actual column names: {list(features_df.columns)}"

        # Test values in 'Sentiment' column should be either 0 or 1
        sentiment_values = features_df['Sentiment'].unique()
        assert all(val in [0, 1] for val in sentiment_values), \
            f"Values in 'Sentiment' column do not meet expectations. Expected values: [0, 1], Actual values: {list(sentiment_values)}"


# if __name__ == "__main__":
#     input_df = pd.read_csv(r"root/unit_testing/testcase/input/test_sa_input.csv")
#     input_df2 = pd.read_csv(r"root/unit_testing/testcase/input/test_sa_input2.csv")
    
#     # Test data with sentiment column
#     test_augment_train()
#     test_sa_preprocess()
#     test_PREPROCESS_FLAIR(input_df)
#     test_SA_PREPROCESS_TEST(input_df)
#     xgb_df = PREPROCESS_RAW(input_df)
#     test_PREPROCESS_XGB(xgb_df.copy())
#     test_SA_PREPROCESS_TRAIN(xgb_df.copy())
#     test_final_full_data_preprocess(xgb_df.copy())

#     # Test data without sentiment column
#     test_PREPROCESS_FLAIR(input_df2)
#     test_SA_PREPROCESS_TEST(input_df2)
#     xgb_df2 = PREPROCESS_RAW(input_df2)
#     test_PREPROCESS_XGB(xgb_df2.copy())
#     print("All Tests Passed")
