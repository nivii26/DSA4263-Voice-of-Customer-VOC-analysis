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
from gensim.models import KeyedVectors

def tm_preprocess(reviewText, min_len=2, max_len=15):
	# Tokenize and Remove stopwords
	tokenizeText = [token for token in simple_preprocess(reviewText, False, min_len, max_len) if token not in STOPWORDS]
	# Lemmatization
	Lemmatizer = WordNetLemmatizer()
	lemmaText = [Lemmatizer.lemmatize(word) for word in tokenizeText]
	return lemmaText

if __name__ == "__main__":

	current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	master_data = pd.DataFrame(columns=["Text"])

	# Load Data
	for file in os.listdir(r"../../data/processed"):
		if file.endswith(".csv"):
			new_data = pd.read_csv(rf"../../data/processed/{file}")
			master_data = pd.concat([master_data, new_data])

	## Separate into sentiments for topic modelling
	positive_data = master_data.loc[master_data["Sentiment"] == "positive", ["Text"]]
	negative_data = master_data.loc[master_data["Sentiment"] == "negative", ["Text"]]
	
	# Pipeline and Process data
	min_len = 2
	max_len = 15
	positive_data["Text"] = positive_data["Text"].apply(lambda x: tm_preprocess(x, min_len, max_len))
	negative_data["Text"] = negative_data["Text"].apply(lambda x: tm_preprocess(x, min_len, max_len))
	# BOW
	pos_dict = corpora.Dictionary(positive_data["Text"])
	pos_corpus = [pos_dict.doc2bow(text) for text in positive_data["Text"]]
	neg_dict = corpora.Dictionary(negative_data["Text"])
	neg_corpus = [neg_dict.doc2bow(text) for text in negative_data["Text"]]
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
	#for i in range(len(pos_tfidf_corpus)):
	#	pos_tfidf_weights = pos_tfidf_corpus[i]
	#	for term_index, weight in pos_tfidf_weights:
	#		print(f"Term '{pos_dict[term_index]}' has TF-IDF weight of {weight}")

	# Embedding for tfidf data using Word2Vec
	# TODO: Separate out the parameters into configuration files
	"""
	size: Dimensionality of the embedding vectors
	window: size of context window
	min_count: minimum required count of the word for it to be included
	"""
	pos_w2v = models.Word2Vec(sentences=pos_tfidf_corpus, vector_size=100, window=5, min_count=1)
	neg_w2v = models.Word2Vec(sentences=neg_tfidf_corpus, vector_size=100, window=5, min_count=1)

	# Store the whole model
	#pos_w2v.save("pos_w2v.model") # pos_w2v = Word2Vec.load("pos_w2v.model")
	#neg_w2v.save("neg_w2v.model") # neg_w2v = Word2Vec.load("neg_w2v.model")

	# Store just the words + their trained embeddings.
	pos_wv = pos_w2v.wv
	pos_wv.save("pos_wv") #pos_wv = KeyedVectors.load("pos_wv", mmap='r')
	neg_wv = neg_w2v.wv
	neg_wv.save("neg_wv") #neg_wv = KeyedVectors.load("neg_wv", mmap='r')

	# Dimension Reduction
	pos_w2v_embeddings = []
	for document in range(len(pos_tfidf_corpus)):
		w2v_embedding = pos_w2v.wv[document]
		pos_w2v_embeddings.append(w2v_embedding)

	neg_w2v_embeddings = []
	for document in range(len(neg_tfidf_corpus)):
		w2v_embedding = neg_w2v.wv[document]
		neg_w2v_embeddings.append(w2v_embedding)

	n_components = 50
	#svd = TruncatedSVD(n_components=n_components)
	pca = PCA(n_components=n_components)
	#svd_pos_embeddings = svd.fit_transform(pos_w2v_embeddings)
	pca_pos_embeddings = pca.fit_transform(pos_w2v_embeddings)
	pd.DataFrame(pca_pos_embeddings).to_csv("../data/tm/POS_PCA_EMBEDDING.csv", index = False, header = False)
	#svd_neg_embeddings = svd.fit_transform(neg_w2v_embeddings)
	pca_neg_embeddings = pca.fit_transform(neg_w2v_embeddings)
	pd.DataFrame(pca_neg_embeddings).to_csv("../data/tm/NEG_PCA_EMBEDDING.csv", index = False, header = False)
	

	# PCA and SVD are both techniques for dimensionality reduction, which can be useful for topic modeling when dealing 
	# with high-dimensional text data. By reducing the dimensionality of the data, it is possible to identify the most 
	# important topics and extract more meaningful insights from the data.

	# In terms of topic modeling, SVD is often preferred over PCA because SVD can provide a more accurate approximation 
	# of the original data. SVD is also more computationally efficient than PCA for large datasets. However, PCA can be 
	# useful when dealing with small datasets or when there is a need for interpretability of the extracted components.
	




	



	    
	