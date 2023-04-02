import pandas as pd
import re
import os
import nltk
import nlpaug.augmenter.word as naw
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import datetime
import joblib

# download necessary NLTK data (only need to run this once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')




def sa_preprocess(text):
    # convert to lowercase
    text = text.lower()
    # remove non-alphabetic characters
    text = re.sub(r'[^a-z]', ' ', text)
    # tokenize the text into words
    tokens = word_tokenize(text)
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def over_sampling(train_data):
    # over sampling
    ros = RandomOverSampler(sampling_strategy='minority')
    X = train_data['Text'].values.reshape(-1, 1)
    y = train_data['Sentiment']
    X_resampled, y_resampled = ros.fit_resample(X, y)
    train_data = pd.DataFrame({'Text': X_resampled.ravel(), 'Sentiment': y_resampled})
    return train_data

# Function to augment negative training data
def augment_train(train_data):
  positive_texts = train_data[train_data['Sentiment'] == 'positive']['Text'].tolist()
  negative_texts = train_data[train_data['Sentiment'] == 'negative']['Text'].tolist()


  # Over-sample negative training examples using nlp-aug
  aug = naw.SynonymAug()
  augmented_texts = aug.augment(negative_texts, n=len(negative_texts))
  negative_labels = ['negative'] * len(negative_texts)
  augmented_df = pd.DataFrame({'Text': augmented_texts, 'Sentiment': negative_labels})

  # Combine Over-sampled examples with train_data
  train_data = pd.concat([train_data, augmented_df], ignore_index=True)

  return train_data

def features_sa_train(train_data):
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

    return features_df, word2vec_model, tfidf, pca_emb, pca_tfidf

def features_sa_test(test_data,word2vec_model, tfidf, pca_emb, pca_tfidf):
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

    # add the label column to the feature matrix DataFrame
    label = features_df.columns
    temp = pd.DataFrame(test_data['Sentiment'].tolist(),columns=["Sentiment"])
    features_df = pd.concat([features_df, temp], axis=1)
    
    return features_df,test_data["Sentiment"]


if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    master_data = pd.DataFrame(columns=["Text"])

    # Load Data
    for file in os.listdir(r"../../data/processed"):
        if file.endswith(".csv"):
            new_data = pd.read_csv(rf"../../data/processed/{file}")
            master_data = pd.concat([master_data, new_data])
    
    # process data and feature engineering
    train_feature, word2vec_model, tfidf, pca_emb, pca_tfidf = features_sa_train(master_data)

    # Saving the model and data
    word2vec_model.save('../../models/w2v_model')
    joblib.dump(tfidf, '../../models/tfidf_sa,pkl')
    joblib.dump(pca_emb, '../../models/pca_emb')
    joblib.dump(pca_tfidf, '../../models/pca_tfidf')
    train_feature.to_csv("../data/sa/features_train_sa.csv", index=False)
