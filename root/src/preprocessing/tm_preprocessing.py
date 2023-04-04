import os
import numpy as np
import pandas as pd
import datetime
from sklearn.decomposition import TruncatedSVD, PCA

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

def tm_preprocess_train(train_data, min_len=2, max_len=15):
	positive_data = train_data.loc[train_data["Sentiment"] == "positive", ["Text"]]
	negative_data = train_data.loc[train_data["Sentiment"] == "negative", ["Text"]]
	positive_data["Text"] = positive_data["Text"].apply(lambda x: tm_preprocess_text(x, min_len, max_len))
	negative_data["Text"] = negative_data["Text"].apply(lambda x: tm_preprocess_text(x, min_len, max_len))
	
	# BOW
	# Create and Save the Dictionary of the Corpus
	pos_dict = corpora.Dictionary(positive_data["Text"])
	pos_corpus = [pos_dict.doc2bow(text) for text in positive_data["Text"]]
	neg_dict = corpora.Dictionary(negative_data["Text"])
	neg_corpus = [neg_dict.doc2bow(text) for text in negative_data["Text"]]

	# TF_IDF
	pos_tfidf = models.TfidfModel(pos_corpus)
	pos_tfidf_corpus = pos_tfidf[pos_corpus]
	neg_tfidf = models.TfidfModel(neg_corpus)
	neg_tfidf_corpus = neg_tfidf[neg_corpus]
	
	# TODO: Move this to Model_Train
	#pos_tfidf.save('../../../models/tm/pos_tfidf')
	#neg_tfidf.save('../../../models/tm/neg_tfidf')
	# Saving the corpus and dict
	"""
	corpus: A list of document vectors, where each vector is represented as a 
	tuple of word IDs and their corresponding frequencies.

	dictionary: A mapping between words and their unique integer IDs in the corpus vocabulary.

	tfidf_model: A term frequency-inverse document frequency (TF-IDF) model that assigns weights to 
	words based on their frequency in the corpus and their relevance to the document.
	"""
	#pos_dict.save('../../../models/tm/pos_dict.dict') # can be loaded back using corpora.dictionary.load(fname)
	#neg_dict.save('../../../models/tm/neg_dict.dict')
	#corpora.MmCorpus.serialize('../../../models/tm/pos_corpus.mm', pos_corpus) # can be loaded back using corpora.MmCorpus(fname)
	#corpora.MmCorpus.serialize('../../../models/tm/neg_corpus.mm', neg_corpus)
	#corpora.MmCorpus.serialize('../../../models/tm/pos_tfidf_corpus.mm', pos_tfidf_corpus)
	#corpora.MmCorpus.serialize('../../../models/tm/neg_tfidf_corpus.mm', neg_tfidf_corpus)
	
	return [pos_dict, neg_dict, pos_corpus, neg_corpus, pos_tfidf, neg_tfidf, pos_tfidf_corpus, neg_tfidf_corpus]

def tm_preprocess_test(test_data, min_len=2, max_len=15):
	test_data["Sentiment"] = test_data["Sentiment"].apply(lambda x: x.lower().strip())
	test_data["Sentiment"] = test_data["Sentiment"].apply(lambda x: x.lower().strip())
	positive_data = test_data.loc[test_data["Sentiment"] == "positive"]
	negative_data = test_data.loc[test_data["Sentiment"] == "negative"]
	positive_data["processed_text"] = positive_data["Text"].apply(lambda x: tm_preprocess_text(x, min_len, max_len))
	negative_data["processed_text"] = negative_data["Text"].apply(lambda x: tm_preprocess_text(x, min_len, max_len))
	return positive_data, negative_data

if __name__ == "__main__":

	os.chdir(r"./root/src/preprocessing")
	current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	master_data = pd.DataFrame(columns=["Sentiment", "Time", "Text"])

	# Load Data
	for file in os.listdir(r"../../data/processed"):
		if file.endswith(".csv"):
			new_data = pd.read_csv(rf"../../data/processed/{file}")
			master_data = pd.concat([master_data, new_data])

	pos_dict, neg_dict, pos_corpus, neg_corpus, pos_tfidf, neg_tfidf, pos_tfidf_corpus, neg_tfidf_corpus = tm_preprocess_train(master_data)

	# Saving the corpus and dict
	"""
	corpus: A list of document vectors, where each vector is represented as a 
	tuple of word IDs and their corresponding frequencies.

	dictionary: A mapping between words and their unique integer IDs in the corpus vocabulary.

	tfidf_model: A term frequency-inverse document frequency (TF-IDF) model that assigns weights to 
	words based on their frequency in the corpus and their relevance to the document.
	"""
	master_data.to_csv("../data/tm/{CURRENT_TIME}_CLEANED_DF.csv", index = False)
	neg_tfidf.save('../../models/tm/neg_tfidf')
	pos_tfidf.save('../../models/tm/pos_tfidf')
	pos_dict.save('../../models/tm/pos_dict.dict') # can be loaded back using corpora.dictionary.load(fname)
	neg_dict.save('../../models/tm/neg_dict.dict')
	corpora.MmCorpus.serialize('../../models/tm/pos_corpus.mm', pos_corpus) # can be loaded back using corpora.MmCorpus(fname)
	corpora.MmCorpus.serialize('../../models/tm/neg_corpus.mm', neg_corpus)
	corpora.MmCorpus.serialize('../../models/tm/pos_tfidf_corpus.mm', pos_tfidf_corpus)
	corpora.MmCorpus.serialize('../../models/tm/neg_tfidf_corpus.mm', neg_tfidf_corpus)
	# Print the TF-IDF weights for each term in the document
	#for i in range(len(pos_tfidf_corpus)):
	#	pos_tfidf_weights = pos_tfidf_corpus[i]
	#	for term_index, weight in pos_tfidf_weights:
	#		print(f"Term '{pos_dict[term_index]}' has TF-IDF weight of {weight}")