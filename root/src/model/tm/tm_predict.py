"""Functions relevant to making a prediction"""

from typing import List
import pandas as pd
from gensim.models import LdaModel, Nmf, LsiModel, TfidfModel
from gensim import corpora
from .core import MODEL_DIR, CONFIG, ROOT_DIR


def load_model():
    """Load persisted model object"""
    model = None
    model_path = str(MODEL_DIR / CONFIG["model_file"])
    if CONFIG["model_name"] == "lda":
        model = LdaModel.load(model_path)
    elif CONFIG["model_name"] == "nmf":
        model = Nmf.load(model_path)
    elif CONFIG["model_name"] == "lsa":
        model = LsiModel.load(model_path)
    if model is None:
        raise Exception(
            """Cannot find appropriate model to load. Check
                        model name again"""
        )
    return model


def preprocess(list_of_text: List[str]) -> List[tuple]:
    """Conducts bow preprocessing on a list of text
    Parameters:
        list_of_text: Each embedded list is a document
    Returns:
        corpus (gensim corpus object): Preprocessed document
    """
    bow_dict = corpora.Dictionary.load(str(MODEL_DIR / CONFIG["id2word_file"]))
    corpus = [bow_dict.doc2bow(text) for text in list_of_text]
    if CONFIG["preprocess_type"] == "tfidf":
        tfidf_model = TfidfModel.load(str(MODEL_DIR / CONFIG["tfidf_model_file"]))
        corpus = [converted for converted in tfidf_model[corpus]]
    return corpus


def predict(bow_document: List[tuple], model) -> tuple:
    """Get topic model predictions
    Parameters:
        bow_document (list): A document in the form of [(word id, importance), (word id, importance) ...]
        model (gensim model object): Either lda, nmf, lsa
    Returns:
        mapped_pred (tuple): (topic label, proba)
    """
    pred = model[bow_document]
    topic_map = CONFIG["topic_map"]
    mapped_pred = [
        (topic_map[str(topic_num)], round(proba, 2)) for topic_num, proba in pred
    ]
    return mapped_pred


def batch_predict(corpus: List[tuple], model) -> List[tuple]:
    """Make batch prediction
    Parameters:
        corpus (list): Each embedded list within is in the form [(word id, importance), (word id, importance) ...]
        model (gensim model object): Either lda, nmf, lsa
    Returns:
        (list): Each embedded tuple is of form (topic label, proba)
    """
    return [predict(doc, model) for doc in corpus]


def TM_MODEL_PREDICT(tm_df: pd.DataFrame) -> pd.DataFrame:
    """Load persisted model -> apply preprocessing methods -> predict

    Parameters:
        tm_df (pandas dataframe): Each value in 'processed_text' column should be a list of tokens

    Returns:
        tm_df (pandas dataframe): tm_df appended with 'Predicted Topic'
    """

    tm_df = tm_df.copy()
    model = load_model()
    corpus = preprocess(tm_df["processed_text"].tolist())
    batch_predictions = batch_predict(corpus, model)
    tm_df["Predicted Topic"] = batch_predictions
    return tm_df.drop("processed_text", axis=1)


if __name__ == "__main__":
    from ast import literal_eval
    df = pd.read_csv(
        str(ROOT_DIR / "src" / "data" / "tm" / "20230405230550_CLEANED_DF.csv"),
        nrows=30,
    )
    # print(df)
    df["processed_text"] = df["Text"].apply(lambda x: literal_eval(x))
    print(TM_MODEL_PREDICT(df))
