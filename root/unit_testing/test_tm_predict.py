import pytest
from ..src.model.tm.tm_predict import (
    load_model,
    preprocess,
    predict,
    batch_predict,
    TM_MODEL_PREDICT,
)
import pandas as pd
from ..src.preprocessing.rawdata_preprocessing import PREPROCESS_RAW
from ..src.preprocessing.tm_preprocessing import TM_PREPROCESS_TEST


@pytest.fixture
def sample_documents():
    # model = load_model()
    sentiment = ["positive", "negative"]
    time = ["18/6/21", "29/7/19"]
    text = [
        "This is a very healthy dog food. Good for their digestion.",
        "THis product is definitely not as good as some other gluten free cookies!",
    ]
    df = pd.DataFrame({"Sentiment": sentiment, "Time": time, "Text": text})
    df = PREPROCESS_RAW(df)
    df = TM_PREPROCESS_TEST(df)
    return df


def test_predict(sample_documents):
    corpus = preprocess([sample_documents["processed_text"][0]])[0]
    model = load_model()
    prediction = predict(corpus, model)
    assert isinstance(prediction, list)
    flag = True
    for pred in prediction:
        if not isinstance(pred[0], str) or not isinstance(pred[1], float):
            flag = False
    assert flag is True


def test_batch_predict(sample_documents):
    corpus = preprocess(sample_documents["processed_text"].tolist())
    model = load_model()
    batch_predictions = batch_predict(corpus, model)
    assert isinstance(batch_predictions, list)
    assert len(batch_predictions) == 2
    flag = True
    for prediction in batch_predictions:
        for pred in prediction:
            if not isinstance(pred[0], str) or not isinstance(pred[1], float):
                flag = False
    assert flag is True


def test_tm_model_predict(sample_documents):
    df_pred = TM_MODEL_PREDICT(sample_documents)
    assert df_pred.columns.tolist() == ["Sentiment", "Time", "Text", "Predicted Topic"]
    assert df_pred.shape == (2, 4)
