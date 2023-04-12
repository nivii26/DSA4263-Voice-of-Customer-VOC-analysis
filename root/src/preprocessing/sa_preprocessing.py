# Basic requirements
import pandas as pd
import numpy as np
import os
import random
import datetime

# for saving models
import joblib

# For pre-processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from .rawdata_preprocessing import PREPROCESS_RAW, remove_html

# download necessary NLTK data (only need to run this once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def sa_preprocess(text):
	'''
    input : raw text DataFrame
    output : tokens
    function : tokenize and other initial processing of text data
    '''
	# tokenize the text into words
	tokens = word_tokenize(text)
	# remove stopwords
	stop_words = set(stopwords.words('english'))
	tokens = [word for word in tokens if word not in stop_words]
	# lemmatize the words
	lemmatizer = WordNetLemmatizer()
	tokens = [lemmatizer.lemmatize(word) for word in tokens]
	return tokens


def augment_train(train_data):
	'''
    input : train_data DataFrame
    output : augumented train_data
    function : augment negative training data
    '''

	positive_texts = train_data[train_data['Sentiment'] == 'positive']['Text'].tolist()
	negative_texts = train_data[train_data['Sentiment'] == 'negative']['Text'].tolist()
	# Over-sample negative training examples using nlp-aug
	random.seed(4263)
	aug = naw.SynonymAug()
	augmented_texts = aug.augment(negative_texts, n=len(negative_texts))
	negative_labels = ['negative'] * len(negative_texts)
	augmented_df = pd.DataFrame({'Text': augmented_texts, 'Sentiment': negative_labels})
	# Combine Over-sampled examples with train_data
	train_data = pd.concat([train_data, augmented_df], ignore_index=True)
	return train_data

def PREPROCESS_XGB(test_data):
	"""
	Input: test_data(dataframe)-cleaned data
	Output: processed test data 
	function: preprocess test data for xgb and other non-transformer based models
	"""

	if 'Sentiment' in test_data.columns.to_list():
		s = test_data['Sentiment']

	# load model for test data
	word2vec_model = Word2Vec.load('root/models/sa/w2v_model')
	tfidf = joblib.load('root/models/sa/tfidf_sa.pkl')
	pca_emb = joblib.load('root/models/sa/pca_emb.pkl')
	pca_tfidf = joblib.load('root/models/sa/pca_tfidf.pkl')

	# apply the preprocessing function to the text data
	test_data['Text'] = test_data['Text'].apply(sa_preprocess)

	## Features
	# train a Word2Vec model on the preprocessed text data
	test_embeddings = test_data['Text'].apply(lambda x: np.mean([word2vec_model.wv[Text] for Text in x if Text in word2vec_model.wv.key_to_index], axis=0))

	# create a new DataFrame for the feature matrix
	features_df = pd.DataFrame(test_embeddings.tolist(), index=test_embeddings.index)

	# perform PCA with n_components set to retain 98% of variance
	features_emb_pca = pca_emb.transform(features_df)

	# create a new DataFrame for the PCA features
	pca_emb_cols = [f"PC_emb{i+1}" for i in range(features_emb_pca.shape[1])]
	pca_df_emb = pd.DataFrame(features_emb_pca, columns=pca_emb_cols)
	

	# obtain the TF-IDF feature matrix for the training and test data
	test_matrix = tfidf.transform(test_data['Text'].apply(lambda x: ' '.join(x))).toarray()
	tfidf_features_df = pd.DataFrame(test_matrix, columns=tfidf.get_feature_names_out())

	# perform PCA with n_components set to retain 95% of variance
	features_tfidf_pca = pca_tfidf.transform(tfidf_features_df)

	# create a new DataFrame for the PCA features
	pca_tfidf_cols = [f"PC_tfidf{i+1}" for i in range(features_tfidf_pca.shape[1])]
	pca_df_tfidf = pd.DataFrame(features_tfidf_pca, columns=pca_tfidf_cols)

	# add the TF-IDF features to the feature matrix DataFrame
	features_df = pd.concat([pca_df_tfidf, pca_df_emb], axis=1)
	
	if 'Sentiment' in test_data.columns.to_list():
		features_df['Sentiment'] = np.array(s)
		features_df['Sentiment'] = features_df['Sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

	return features_df

def PREPROCESS_FLAIR(raw_data):
	"""
	Input: raw_data(dataframe)
	Output: final_cleaned_data_flair(dataframe)
	function: preprocess test data for flair and other transformer based models
	"""
	if 'Sentiment' in raw_data.columns:
		final_cleaned_data_flair = pd.DataFrame(columns=["Sentiment", "Time", "Text"])
	else:
		final_cleaned_data_flair = pd.DataFrame(columns=["Time", "Text"])
	
	## Clean the data
	cleaned_data_flair = raw_data.dropna().drop_duplicates()
	## Preprocess the Review Column
	cleaned_data_flair["Text"] = cleaned_data_flair["Text"].apply(remove_html)
	## Combine all the cleaned datasets
	final_cleaned_data_flair = pd.concat([final_cleaned_data_flair, cleaned_data_flair])

	
	if 'Sentiment' in raw_data.columns:
		final_cleaned_data_flair['Sentiment'] = final_cleaned_data_flair['Sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
	
    
	return final_cleaned_data_flair


def SA_PREPROCESS_TRAIN(train_data):
	"""
	Input: train_data(dataframe) - cleaned data
	Output: features_df(dataframe), SAVE : word2vec_model, tfidf, pca_emb, pca_tfidf 
	function: preprocess training data 
	"""
	# apply the augmentation function to the preprocessed text data
	train_data = augment_train(train_data)
	print("Shape after augmenting negative training samples: ", train_data.shape)

	# apply the preprocessing function to the text data
	train_data['Text'] = train_data['Text'].apply(sa_preprocess)

	## Features
	# train a Word2Vec model on the preprocessed text data
	word2vec_model = Word2Vec(train_data['Text'], min_count=1)
	train_embeddings = train_data['Text'].apply(lambda x: np.mean([word2vec_model.wv[Text] for Text in x if Text in word2vec_model.wv.key_to_index], axis=0))

	# create a new DataFrame for the feature matrix
	features_df = pd.DataFrame(train_embeddings.tolist(), index=train_embeddings.index)

	# perform PCA with n_components set to retain 98% of variance
	pca_emb = PCA(n_components=0.98)
	pca_emb.fit(features_df)
	features_emb_pca = pca_emb.transform(features_df)

	# create a new DataFrame for the PCA features
	pca_emb_cols = [f"PC_emb{i+1}" for i in range(features_emb_pca.shape[1])]
	pca_df_emb = pd.DataFrame(features_emb_pca, columns=pca_emb_cols)


	# create a TF-IDF vectorizer object
	tfidf = TfidfVectorizer()
	
	# fit the vectorizer on the preprocessed text data
	tfidf.fit(train_data['Text'].apply(lambda x: ' '.join(x)))

	# obtain the TF-IDF feature matrix for the training and test data
	train_matrix = tfidf.transform(train_data['Text'].apply(lambda x: ' '.join(x))).toarray()
	tfidf_features_df = pd.DataFrame(train_matrix, columns=tfidf.get_feature_names_out())
	# perform PCA with n_components set to retain 95% of variance
	pca_tfidf = PCA(n_components=0.95)
	pca_tfidf.fit(tfidf_features_df)
	features_tfidf_pca = pca_tfidf.transform(tfidf_features_df)
	
	# create a new DataFrame for the PCA features
	pca_tfidf_cols = [f"PC_tfidf{i+1}" for i in range(features_tfidf_pca.shape[1])]
	pca_df_tfidf = pd.DataFrame(features_tfidf_pca, columns=pca_tfidf_cols)

	# add the TF-IDF features to the feature matrix DataFrame
	features_df = pd.concat([pca_df_tfidf, pca_df_emb], axis=1)

	# add the label column to the feature matrix DataFrame
	label = features_df.columns
	features_df['Sentiment'] = train_data['Sentiment']
	features_df['Sentiment'] = features_df['Sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
	
	# Saving the model and data
	word2vec_model.save('root/models/sa/w2v_model')
	joblib.dump(tfidf, 'root/models/sa/tfidf_sa.pkl')
	joblib.dump(pca_emb, 'root/models/sa/pca_emb.pkl')
	joblib.dump(pca_tfidf, 'root/models/sa/pca_tfidf.pkl')
	# features_df.to_csv("../data/sa/features_train_sa.csv", index=False)
	return features_df

def SA_PREPROCESS_TEST(raw_data):
	"""
	Input: raw_data(dataframe)
	Output: SA_PROCESSED_DF_XGB, SA_PROCESSED_DF_FLAIR
	function: preprocess test data for two kinds of models seperately 
	"""
	cleaned_data = PREPROCESS_RAW(raw_data)
	SA_PROCESSED_DF_XGB = PREPROCESS_XGB(cleaned_data)
	SA_PROCESSED_DF_FLAIR = PREPROCESS_FLAIR(raw_data)
	return SA_PROCESSED_DF_XGB, SA_PROCESSED_DF_FLAIR

def final_full_data_preprocess(test_data):

	# Load the model
	# load model for test data
	word2vec_model = Word2Vec.load('root/models/sa/w2v_model')
	tfidf = joblib.load('root/models/sa/tfidf_sa.pkl')
	pca_emb = joblib.load('root/models/sa/pca_emb.pkl')
	pca_tfidf = joblib.load('root/models/sa/pca_tfidf.pkl')

	# apply the augmentation function to the preprocessed text data
	test_data = augment_train(test_data)
	print("Shape after augmenting negative training samples: ", test_data.shape)

	# apply the preprocessing function to the text data
	test_data['Text'] = test_data['Text'].apply(sa_preprocess)

	## Features
	# train a Word2Vec model on the preprocessed text data
	test_embeddings = test_data['Text'].apply(lambda x: np.mean([word2vec_model.wv[Text] for Text in x if Text in word2vec_model.wv.key_to_index], axis=0))

	# create a new DataFrame for the feature matrix
	features_df = pd.DataFrame(test_embeddings.tolist(), index=test_embeddings.index)

	# perform PCA with n_components set to retain 98% of variance
	features_emb_pca = pca_emb.transform(features_df)

	# create a new DataFrame for the PCA features
	pca_emb_cols = [f"PC_emb{i+1}" for i in range(features_emb_pca.shape[1])]
	pca_df_emb = pd.DataFrame(features_emb_pca, columns=pca_emb_cols)
	
	# obtain the TF-IDF feature matrix for the training and test data
	test_matrix = tfidf.transform(test_data['Text'].apply(lambda x: ' '.join(x))).toarray()
	tfidf_features_df = pd.DataFrame(test_matrix, columns=tfidf.get_feature_names_out())

	# perform PCA with n_components set to retain 95% of variance
	features_tfidf_pca = pca_tfidf.transform(tfidf_features_df)

	# create a new DataFrame for the PCA features
	pca_tfidf_cols = [f"PC_tfidf{i+1}" for i in range(features_tfidf_pca.shape[1])]
	pca_df_tfidf = pd.DataFrame(features_tfidf_pca, columns=pca_tfidf_cols)

	# add the TF-IDF features to the feature matrix DataFrame
	features_df = pd.concat([pca_df_tfidf, pca_df_emb], axis=1)

	features_df['Sentiment'] = test_data['Sentiment']
	features_df['Sentiment'] = features_df['Sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
	
	return features_df



if __name__ == "__main__":

	os.chdir("./root/src/preprocessing")
	current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	master_data = pd.DataFrame(columns=["Sentiment", "Time", "Text"])

	# Load Data
	for file in os.listdir(r"../../data/processed"):
		if file.endswith(".csv"):
			new_data = pd.read_csv(rf"../../data/processed/{file}")
			master_data = pd.concat([master_data, new_data])
	
	# process data and feature engineering for training data
	train_feature = SA_PREPROCESS_TRAIN(master_data)

	# process data and feature engineering for test data
	#SA_PROCESSED_DF_XGB, SA_PROCESSED_DF_FLAIR=SA_PREPROCESS_TEST(raw_data)
	#SA_PROCESSED_DF_XGB.to_csv("../data/sa/features_train_sa_new.csv", index=False)
	#SA_PROCESSED_DF_FLAIR.to_csv("../data/sa/features_test_sa_new.csv", index=False)
