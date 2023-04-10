import pytest
from unittest import TestCase
from gensim.models import LdaModel, Nmf, LsiModel
from ..src.model.tm.tm_train import (
    build_gensim_model, 
    compute_coherence_score,
    evaluate_topic_models)
import pandas as pd
from ..src.preprocessing.rawdata_preprocessing import PREPROCESS_RAW
from ..src.preprocessing.tm_preprocessing import TM_PREPROCESS_TRAIN


@pytest.fixture
def generate_variables():
    sentiment = ['positive', 'negative']
    time = ['18/6/21', '29/7/19']
    text = [
        'This is a very healthy dog food. Good for their digestion.',
        'THis product is definitely not as good as some other gluten free cookies!'
    ]
    df = pd.DataFrame({
        'Sentiment': sentiment,
        'Time': time,
        'Text': text
        })
    df = PREPROCESS_RAW(df)
    bow_dict, bow_corpus, tfidf_model, tfidf_corpus = TM_PREPROCESS_TRAIN(df)
    lemma_text = [[bow_dict[bow[0]] for bow in sent] for sent in bow_corpus]
    num_topics = [2]
    return bow_dict, bow_corpus, tfidf_model, tfidf_corpus, lemma_text, num_topics


class BuildGensimModelTests(TestCase):    
    @pytest.fixture(autouse=True)
    def init_vars(self, generate_variables):
        bow_dict, bow_corpus, tfidf_model, tfidf_corpus, \
            lemma_text, num_topics = generate_variables
        self.bow_dict = bow_dict
        self.bow_corpus = bow_corpus
        self.tfidf_model = tfidf_model
        self.tfidf_corpus = tfidf_corpus
        self.lemma_text = lemma_text
        self.num_topics = num_topics
    
    def test_build_lda_model(self):
        model = build_gensim_model(
            'lda', self.bow_corpus, self.bow_dict, self.num_topics[0])
        assert isinstance(model, LdaModel)

    def test_build_nmf_model(self):
        model = build_gensim_model(
            'nmf', self.bow_corpus, self.bow_dict, self.num_topics[0])
        assert isinstance(model, Nmf)

    def test_build_lsa_model(self):
        model = build_gensim_model(
            'lsa', self.bow_corpus, self.bow_dict, self.num_topics[0])
        assert isinstance(model, LsiModel)
    
    def test_build_non_gensim_model(self):
        with pytest.raises(ValueError):
            build_gensim_model('bert', self.bow_corpus, self.bow_dict, self.num_topics[0]) 


@pytest.mark.parametrize(
    'model_name', ['lda', 'nmf', 'lsa']
)
def test_compute_coherence_score(generate_variables, model_name):
    bow_dict, bow_corpus, _, _, \
        lemma_text, num_topics = generate_variables
    score = compute_coherence_score(
        model_name, bow_corpus, bow_dict, lemma_text, num_topics[0])
    assert isinstance(score, float)


def test_evaluate_topic_models(generate_variables):
    bow_dict, bow_corpus, tfidf_model, tfidf_corpus, \
        lemma_text, num_topics = generate_variables
    model_names = ['lda', 'nmf', 'lsa']
    corpuses = {
        'bow': bow_corpus,
        'tfidf': tfidf_corpus
    }
    results = evaluate_topic_models(
        model_names, corpuses, bow_dict, lemma_text, num_topics)
    assert results.shape == (6, 4)
    