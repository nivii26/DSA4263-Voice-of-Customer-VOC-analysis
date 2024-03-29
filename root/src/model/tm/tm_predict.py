"""Functions relevant to making a prediction"""

from typing import List, Tuple
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


def preprocess(list_of_text: List[List[str]]) -> List[List[tuple]]:
    """Conducts bow preprocessing on a list of text
    Parameters:
        list_of_text (list): Each embedded list is a document
    Returns:
        corpus (list): Each embedded list contains tuples in the form (word id, embedded value)
    """
    bow_dict = corpora.Dictionary.load(str(MODEL_DIR / CONFIG["id2word_file"]))
    corpus = [bow_dict.doc2bow(text) for text in list_of_text]
    if CONFIG["preprocess_type"] == "tfidf":
        tfidf_model = TfidfModel.load(str(MODEL_DIR / CONFIG["tfidf_model_file"]))
        corpus = [converted for converted in tfidf_model[corpus]]
    return corpus


def predict(bow_document: List[tuple], model) -> List[tuple]:
    """Get topic model predictions
    Parameters:
        bow_document (list): A document in the form of [(word id, importance), (word id, importance) ...]
        model (gensim model object): Either lda, nmf, lsa
    Returns:
        mapped_pred (list): Each embedded tuple is of the form (topic label, proba)
    """
    pred = model[bow_document]
    topic_map = CONFIG["topic_map"]
    mapped_pred = [
        (topic_map[str(topic_num)], round(proba, 3)) for topic_num, proba in pred
    ]
    return mapped_pred


def batch_predict(corpus: List[List[tuple]], model) -> List[List[tuple]]:
    """Make batch prediction
    Parameters:
        corpus (list): Each embedded list within is in the form [(word id, importance), (word id, importance) ...]
        model (gensim model object): Either lda, nmf, lsa
    Returns:
        (list): Each embedded list contains tuples of form (topic label, proba)
    """
    return [predict(doc, model) for doc in corpus]

def extract_topic(row: List[Tuple[str, float]]) -> str:
    """Get top topic for each review
    Parameters:
        row (List[Tuple[str, float]]): Each row of data from a dataframe
    Returns:
        (str): A string that contains the main topic that the review is about 
    """
    topics = row['Predicted Topic']
    max_topic = max(topics, key=lambda x: x[1])
    return max_topic[0]

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
    tm_df['Main Topic'] = tm_df.apply(extract_topic, axis=1)
    return tm_df