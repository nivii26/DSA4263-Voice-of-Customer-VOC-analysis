# modelling
from gensim.models import LdaModel, Nmf, LsiModel, CoherenceModel
from tqdm import tqdm
from pathlib import Path

# plotting and data manipulation
import pandas as pd


def build_topic_model(model_name, corpus, id2word, hyperparameters={}, random_state=0):
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
        model = LdaModel(id2word=id2word, corpus=corpus, random_state=random_state, **hyperparameters)
    elif model_name == 'nmf':
        model = Nmf(id2word=id2word, corpus=corpus, random_state=random_state, **hyperparameters)
    elif model_name == 'lsa':
        model = LsiModel(id2word=id2word, corpus=corpus, random_seed=random_state, **hyperparameters)
    # elif model_name == 'hdp':
    #     hyperparameters["T"] = hyperparameters.pop("num_topics")
    #     model = HdpModel(id2word=id2word, corpus=corpus, random_state=random_state, **hyperparameters)
    else:
        model = None
    return model

def compute_coherence_score(model_name, corpus, id2word, lemma_text, hyperparameters={}, random_state=0, coherence_metric='c_v'):
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
    model = build_topic_model(model_name, corpus, id2word, hyperparameters=hyperparameters, random_state=random_state)
    if model is None:
        return 0.0 # or raise an exception
    coherence_model_lda = CoherenceModel(model=model, texts=lemma_text, dictionary=id2word, coherence=coherence_metric)
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
                score = compute_coherence_score(name, corpus, id2word, lemma_text, hyperparameters={'num_topics': k})
                
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

def predict(model, processed_text, pred_map):
    """Get topic model predictions 
    Parameters:
        model (gensim topic model object): model for making predictions
        processed_text (list of tuples): output from a bag of words or tfidf gensim conversion 
        pred_map (dict): map numerical topic to topic label
    
    Returns:
        mapped_pred (list of tuples): [(topic label 1, proba), (topic label 2, proba)...]
    """
    pred = model[processed_text]
    mapped_pred = [(pred_map[pred_topic], str(round(proba, 2))) for pred_topic, proba in pred]
    return mapped_pred

def tm_model_predict(processed_df, sentiment):
    """Loads persisted model and uses the predict function to get topic model predictions

    Parameters:
        processed_df (pandas dataframe): format is same as to reviews dataframe, just with 
        an added column denoting the preprocessed version of the text
        sentiment (str): 'Positive' or 'Negative'
    
    Returns:
        (pandas dataframe): Dataframe in the same format as the reviews dataframe, with an 
        added columns denoting the predicted topics 
    """
    sentiment = sentiment.lower().strip()

    if sentiment not in ['positive', 'negative']:
        raise Exception('Sentiment should either be positive or negative')
    
    sentiment_map = {
        'positive': {
            0: 'Food condiments'
            1: 'Product quality and affordability'
            2: 'Tea flavors and preparation'
            3: 'Drinks and flavors'
            4: 'Positive product reviews'
            5: 'Food products and nutrition'
            6: 'Healthy snack options'
            7: 'Product value and satisfaction'
            8: 'Coffee flavors and preferences'
            9: 'Preferences'
            10: 'Pet food'
        }, 
        'negative': {
            0: 'Beverage',
            1: 'Pet products'
        }
    }

    MODEL_SAVE = Path(__file__).parent.parent.parent / 'models'

    if sentiment == 'positive':
        model = Nmf.load(f'{MODEL_SAVE}/tm_pos_model')
    else:
        model = LsiModel.load(f'{MODEL_SAVE}/tm_neg_model')
    
    processed_df['Predicted Topic'] = (
        processed_df['Preprocessed Text']
        .apply(lambda x: predict(model, x, sentiment_map[sentiment]))
    )
    
    return processed_df.drop('Preprocessed Text', axis=1)