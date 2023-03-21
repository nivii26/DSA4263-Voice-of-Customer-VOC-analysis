# modelling
from gensim.models.ldamodel import LdaModel
from gensim.models.nmf import Nmf
from gensim.models import CoherenceModel
from tqdm import tqdm

# plotting and data manipulation
import pandas as pd

def build_topic_model(model_name, corpus, id2word, 
                      hyperparameters={}, random_state=0):
    """Build a model and evaluate coherence
    
    Parameters:
        model_name (string): name of model to used. lda or nmf here.
        corpus (gensim corpus object): gensim corpus
        id2word (gensim dictionary object): gensim dictionary that contains
        word mappings for words in corpus
        lemma_text (list): list of lists. Each embedded list is a list of all the lemmatised words 
        in a sample
        hyperparameters (dict): dictionary containing hyperparameters for the models used
        random_state (int): random seed
        coherence_metric (string): inputs follow gensim's coherence metrics
    
    Returns:
        (int): coherence score
    """
    
    if model_name == 'lda':
        model = LdaModel(id2word=id2word, 
                         corpus=corpus,
                         random_state=random_state,
                         **hyperparameters)
        
    elif model_name == 'nmf':
        model = Nmf(id2word=id2word,
                    corpus=corpus,
                    random_state=random_state,
                    **hyperparameters)
        
    return model

def compute_coherence_score(model_name, corpus, id2word, lemma_text, 
                            hyperparameters={}, random_state=0, coherence_metric='c_v'):
    """Build a model and evaluate coherence
    
    Parameters:
        model_name (string): name of model to used. lda or nmf here.
        corpus (gensim corpus object): gensim corpus
        id2word (gensim dictionary object): gensim dictionary that contains
        word mappings for words in corpus
        lemma_text (list): list of lists. Each embedded list is a list of all the lemmatised words 
        in a sample
        hyperparameters (dict): dictionary containing hyperparameters for the models used
        random_state (int): random seed
        coherence_metric (string): inputs follow gensim's coherence metrics
    
    Returns:
        (int): coherence score
    """    
    model = build_topic_model(model_name, corpus, id2word, 
                              hyperparameters=hyperparameters, random_state=random_state)
    
    coherence_model_lda = CoherenceModel(model=model, texts=lemma_text, 
                                         dictionary=id2word, coherence=coherence_metric)
    
    return coherence_model_lda.get_coherence()

def evaluate_topic_models(model_name, corpuses, num_topics, id2word, lemma_text):
    """Run different algorithms with different corpus inputs 
    and number of topics.
    
    Parameters:
    model_name (list): collection of different algorithm names from gensim
    corpuses (dictionary): indicate different corpuses to try out. 
    key indicates type of corpus, value indicates reference to gensim corpus object
    num_topics (list): range of number of topics to try topic modelling with
    id2word (gensim dictionary object): gensim dictionary that contains
    word mappings for words in corpus
    lemma_text (list): list of lists. Each embedded list is a list of all the lemmatised 
    words in a sample
    
    Returns:
    Dataframe that records coherence score for each corpus + num_topic combination
    """
    
    # collect results here
    select_algo_results = {
        'model name': [],
        'corpus': [],
        'num topic': [],
        'score': []
    }

    pbar = tqdm(total=len(model_name)*len(corpuses)*len(num_topics))
    for name in model_name:
        for corpus_type, corpus in corpuses.items():
            for k in num_topics:
                score = compute_coherence_score(name, corpus, id2word, lemma_text, 
                                                hyperparameters={'num_topics': k})
                # add model results to dataframe
                select_algo_results['model name'].append(name)
                select_algo_results['corpus'].append(corpus_type)
                select_algo_results['num topic'].append(k)
                select_algo_results['score'].append(score)
                
                # update progress bar
                pbar.update(1)
                
    pbar.close()
    
    return pd.DataFrame(select_algo_results)

def build_top_model(df, corpuses, id2word, random_state=0):
    """Build top model based on coherence score experiment
    
    Parameters:
        df (dataframe): dataframe indicating topic model experiments
        corpuses (dictionary): indicate different corpuses to try out. 
        id2word (gensim dictionary object): gensim dictionary that contains
        word mappings for words in corpus
        random_state (int): random seed
        coherence_metric (string): inputs follow gensim's coherence metrics
    
    Returns:
        (int): coherence score
    """
    # get hyperparameters for top model
    top_model_config = df.loc[df['score']==df['score'].max()]
    
    # extract various values
    top_model_name = top_model_config['model name'].values[0]
    top_corpus = top_model_config['corpus'].values[0]
    top_num_topic = top_model_config['num topic'].values[0]
    
    # build top model
    top_model = build_topic_model(top_model_name, corpuses[top_corpus], id2word, 
                                  hyperparameters={'num_topics': top_num_topic}, 
                                  random_state=random_state)
    return top_model