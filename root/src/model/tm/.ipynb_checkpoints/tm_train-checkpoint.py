"""Contains functions related to training topic models"""

from typing import List
from gensim.models import LdaModel, Nmf, LsiModel, CoherenceModel
from gensim import corpora
from tqdm import tqdm
import pandas as pd
from .core import MODEL_DIR, CONFIG


def build_gensim_model(
    model_name: str,
    corpus: corpora.MmCorpus,
    id2word: corpora.Dictionary,
    num_topics: int,
    random_state=0,
):
    """Build a model from gensim

    Parameters:
        model_name (string): Name of model to build. lda or nmf or lsa.
        corpus (gensim corpus): Bag of words based corpus
        id2word (gensim dictionary object): gensim dict, contains word mappings
        num_topics (int): Number of topics
        random_state (int): Random seed

    Returns:
        (gensim model): trained gensim model
    """
    if model_name == "lda":
        model = LdaModel(
            id2word=id2word,
            corpus=corpus,
            num_topics=num_topics,
            random_state=random_state,
        )
    elif model_name == "nmf":
        model = Nmf(
            id2word=id2word,
            corpus=corpus,
            num_topics=num_topics,
            random_state=random_state,
        )
    elif model_name == "lsa":
        model = LsiModel(
            id2word=id2word,
            corpus=corpus,
            num_topics=num_topics,
            random_seed=random_state,
        )
    else:
        raise ValueError("Model name should be one of lda, nmf, lsa!")
    return model


def compute_coherence_score(
    model_name: str,
    corpus: corpora.MmCorpus,
    id2word: corpora.Dictionary,
    lemma_text: List[str],
    num_topics: int,
    random_state=0,
    coherence_metric="c_v",
) -> int:
    """Build a gensim model and evaluate coherence

    Parameters:
        model_name (string): Name of model to used. lda or nmf or lsa.
        corpus (gensim corpus object): Bag of words based corpus
        id2word (gensim dictionary object): gensim dict, contains word mappings
        lemma_text (list): List containing lists of all the lemmatised words per sample
        num_topics (int): Number of topics
        random_state (int): Random seed
        coherence_metric (string): A gensim coherence metric

    Returns:
        (int): coherence score
    """
    model = build_gensim_model(
        model_name, corpus, id2word, num_topics, random_state=random_state
    )

    coherence_model_lda = CoherenceModel(
        model=model, texts=lemma_text, dictionary=id2word, coherence=coherence_metric
    )

    return coherence_model_lda.get_coherence()


def evaluate_topic_models(
    model_name: List[str],
    corpuses: dict,
    id2word: corpora.Dictionary,
    lemma_text: List[str],
    num_topics: List[int],
) -> pd.DataFrame:
    """Run different gensim algorithms with different corpus inputs
    and number of topics.

    Parameters:
    model_name (list): Collection of different algorithm names from gensim
    corpuses (dictionary): Key - 'bow' or 'tfidf'. Value - gensim corpus object
    id2word (gensim dictionary object): gensim dict, contains word mappings
    lemma_text (list): List containing lists of all the lemmatised words per sample
    num_topics (list): Range of number of topics to try topic modelling with

    Returns:
        (pd.DataFrame) Dataframe with coherence score for each corpus + num_topic combination
    """

    # collect results here
    select_algo_results = {"model name": [], "corpus": [], "num topic": [], "score": []}

    pbar = tqdm(total=len(model_name) * len(corpuses) * len(num_topics))

    for name in model_name:
        for corpus_type, corpus in corpuses.items():
            for k in num_topics:
                score = compute_coherence_score(name, corpus, id2word, lemma_text, k)

                # add model results to dataframe
                select_algo_results["model name"].append(name)
                select_algo_results["corpus"].append(corpus_type)
                select_algo_results["num topic"].append(k)
                select_algo_results["score"].append(score)

                # update progress bar
                pbar.update(1)

    pbar.close()
    return pd.DataFrame(select_algo_results)


def train_and_persist_tm() -> None:
    """Train the selected topic model"""

    if CONFIG["preprocess_type"] == "bow":
        corpus = corpora.MmCorpus(str(MODEL_DIR / CONFIG["bow_corpus_file"]))
    else:
        corpus = corpora.MmCorpus(str(MODEL_DIR / CONFIG["tfidf_corpus_file"]))
    id2word = corpora.Dictionary.load(str(MODEL_DIR / CONFIG["id2word_file"]))
    num_topics = int(CONFIG["num_topics"])
    model_name = CONFIG["model_name"]
    model = build_gensim_model(model_name, corpus, id2word, num_topics)
    model.save(str(MODEL_DIR / "tm_model"))


if __name__ == "__main__":
    train_and_persist_tm()
