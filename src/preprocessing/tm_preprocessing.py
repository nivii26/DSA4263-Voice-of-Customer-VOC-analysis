import os
import numpy as np
import pandas as pd
import datetime

# Other NLP Libraries
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

# Gensim
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

def tm_preprocess(reviewText, min_len=2, max_len=15):
	# Tokenize and Remove stopwords
	tokenizeText = [token for token in simple_preprocess(reviewText, False, min_len, max_len) if token not in STOPWORDS]
	# Lemmatization
	Lemmatizer = WordNetLemmatizer()
	lemmaText = [Lemmatizer.lemmatize(word) for word in tokenizeText]
	return lemmaText

if __name__ == "__main__":
    
	positive_text = pd.DataFrame(columns=["Text"])
	negative_text = pd.DataFrame(columns=["Text"])

    # Load Data
	for file in os.listdir(r"../../data/processed"):
		if file.endswith("TM_POS_DATA.csv"):
			new_data = pd.read_csv(rf"../../data/processed/{file}").loc[:, ["Text"]]
			positive_text = pd.concat([positive_text, new_data])
		elif file.endswith("TM_NEG_DATA.csv"):
			new_data = pd.read_csv(rf"../../data/processed/{file}").loc[:, ["Text"]]
			negative_text = pd.concat([negative_text, new_data])
	
	# Pipeline and Process data
	min_len = 2
	max_len = 15
	positive_text["Text"] = positive_text["Text"].apply(lambda x: tm_preprocess(x, min_len, max_len), axis = 1)
	negative_text["Text"] = negative_text["Text"].apply(lambda x: tm_preprocess(x, min_len, max_len), axis = 1)
	# BOW
	pos_dict = corpora.Dictionary(positive_text["Text"])
	pos_corpus = [pos_dict.doc2bow(text) for text in positive_text["Text"]]
	neg_dict = corpora.Dictionary(negative_text["Text"])
	neg_corpus = [neg_dict.doc2bow(text) for text in negative_text["Text"]]
	# TF_IDF
	pos_tfidf_corpus = models.TfidfModel(pos_corpus)[pos_corpus]
	neg_tfidf_corpus = models.TfidfModel(neg_corpus)[neg_corpus]
	
	# Saving the corpus and dict
	"""
	corpus: A list of document vectors, where each vector is represented as a 
	tuple of word IDs and their corresponding frequencies.

	dictionary: A mapping between words and their unique integer IDs in the corpus vocabulary.

	tfidf_model: A term frequency-inverse document frequency (TF-IDF) model that assigns weights to 
	words based on their frequency in the corpus and their relevance to the document.
	"""
	pos_dict.save('pos_dict.dict') # can be loaded back using corpora.dictionary.load(fname)
	neg_dict.save('neg_dict.dict')
	corpora.MmCorpus.serialize('pos_corpus.mm', pos_corpus) # can be loaded back using corpora.MmCorpus(fname)
	corpora.MmCorpus.serialize('neg_corpus.mm', neg_corpus)
	corpora.MmCorpus.serialize('pos_tfidf_corpus.mm', pos_tfidf_corpus)
	corpora.MmCorpus.serialize('neg_tfidf_corpus.mm', neg_tfidf_corpus)
	# Print the TF-IDF weights for each term in the document
	"""
	for i in range(len(pos_tfidf_corpus)):
		pos_tfidf_weights = pos_tfidf_corpus[i]
		for term_index, weight in pos_tfidf_weights:
			print(f"Term '{pos_dict[term_index]}' has TF-IDF weight of {weight}")
	"""

	



            
        