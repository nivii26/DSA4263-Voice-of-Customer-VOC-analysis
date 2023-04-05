import os
import numpy as np
import pandas as pd
import datetime

# Other NLP Libraries
import nltk
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
#nltk.download('omw-1.4')

# Gensim
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

def tm_preprocess_text(reviewText, min_len=2, max_len=15):
	# Tokenize and Remove stopwords
	tokenizeText = [token for token in simple_preprocess(reviewText, False, min_len, max_len) if token not in STOPWORDS]
	# Lemmatization
	Lemmatizer = WordNetLemmatizer()
	lemmaText = [Lemmatizer.lemmatize(word) for word in tokenizeText]
	return lemmaText

def TM_PREPROCESS_TRAIN(train_data, min_len=2, max_len=15):
	train_data["Text"] = train_data["Text"].apply(lambda x: tm_preprocess_text(x, min_len, max_len))
	
	# BOW
	# Create and Save the Dictionary of the Corpus
	TM_BOW_dict = corpora.Dictionary(train_data["Text"])
	TM_BOW_corpus = [pos_dict.doc2bow(text) for text in train_data["Text"]]

	# TF_IDF
	TM_tfidf = models.TfidfModel(TM_BOW_corpus)
	TM_tfidf_corpus = TM_tfidf[TM_BOW_corpus]
	
	return [TM_BOW_dict, TM_BOW_corpus, TM_tfidf, TM_tfidf_corpus]

def TM_PREPROCESS_TEST(test_data, min_len=2, max_len=15):
	test_data["Sentiment"] = test_data["Sentiment"].apply(lambda x: x.lower().strip())
	test_data["processed_text"] = test_data["Text"].apply(lambda x: tm_preprocess_text(x, min_len, max_len))
	return test_data

if __name__ == "__main__":

	os.chdir(r"./root/src/preprocessing")
	current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	master_data = pd.DataFrame(columns=["Sentiment", "Time", "Text"])

	# Load Data
	for file in os.listdir(r"../../data/processed"):
		if file.endswith(".csv"):
			new_data = pd.read_csv(rf"../../data/processed/{file}")
			master_data = pd.concat([master_data, new_data])

	TM_BOW_dict, TM_BOW_corpus, TM_tfidf, TM_tfidf_corpus = TM_PREPROCESS_TRAIN(master_data)

	# Saving the corpus and dict
	"""
	corpus: A list of document vectors, where each vector is represented as a 
	tuple of word IDs and their corresponding frequencies.

	dictionary: A mapping between words and their unique integer IDs in the corpus vocabulary.

	tfidf_model: A term frequency-inverse document frequency (TF-IDF) model that assigns weights to 
	words based on their frequency in the corpus and their relevance to the document.
	"""
	master_data.to_csv("../data/tm/{CURRENT_TIME}_CLEANED_DF.csv", index = False)
	TM_tfidf.save('../../models/tm/tm_tfidf')
	TM_BOW_dict.save('../../models/tm/tm_bow_dict.dict') # can be loaded back using corpora.dictionary.load(fname)
	corpora.MmCorpus.serialize('../../models/tm/tm_bow_corpus.mm', TM_BOW_corpus) # can be loaded back using corpora.MmCorpus(fname)
	corpora.MmCorpus.serialize('../../models/tm/tm_tfidf_corpus.mm', TM_tfidf_corpus)
	# Print the TF-IDF weights for each term in the document
	#for i in range(len(tm_tfidf_corpus)):
	#	tm_tfidf_weights = tm_tfidf_corpus[i]
	#	for term_index, weight in tm_tfidf_weights:
	#		print(f"Term '{tm_bow_dict[term_index]}' has TF-IDF weight of {weight}")