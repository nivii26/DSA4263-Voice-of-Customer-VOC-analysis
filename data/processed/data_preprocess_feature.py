import pandas as pd
import re
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

# download necessary NLTK data (only need to run this once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# load the data from the csv file
train_data = pd.read_csv("reviews.csv")
# train_label = train_data["Sentiment"]
original_data = pd.DataFrame(train_data["Text"])

# define a function to preprocess the text data
def preprocess_text(text):
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

# remove the html symbol
def remove_html(text):
    regex = r"<[^>]+>"
    text_new = re.sub(regex, " ", text)
    return text_new

# apply the preprocessing function to the text data
train_data['Text'] = train_data['Text'].apply(remove_html)
train_data['Text'] = train_data['Text'].apply(preprocess_text)

# define an NLPAug data augmentation function
def augment_text(text):
    # define an augmentation method
    aug = naw.SynonymAug(aug_src='wordnet', lang='eng')
    # apply the augmentation method to the text
    augmented_text = aug.augment(text)
    return augmented_text

# apply the augmentation function to the preprocessed text data
train_data['Text'] = train_data['Text'].apply(augment_text)

# over sampling
ros = RandomOverSampler(sampling_strategy='minority')
X = train_data['Text'].values.reshape(-1, 1)
y = train_data['Sentiment']
X_resampled, y_resampled = ros.fit_resample(X, y)
train_data = pd.DataFrame({'Text': X_resampled.ravel(), 'Sentiment': y_resampled})

# save the data to a new csv file
train_data.to_csv("oversampling_reviews.csv", index=False)

## Features
# train a Word2Vec model on the preprocessed text data
word2vec_model = Word2Vec(train_data['Text'], min_count=1)

# create a function to generate the word embedding vectors for each sentence
def generate_word_embedding(sentence):
    # initialize an empty array for the sentence vector
    sentence_vector = []
    # loop through each word in the sentence
    for word in sentence:
        try:
            # add the vector representation of the word to the sentence vector
            word_vector = word2vec_model.wv[word]
            sentence_vector.append(word_vector)
        except KeyError:
            # ignore words that are not in the vocabulary
            pass
    # take the mean of the word vectors to get the sentence vector
    sentence_vector = np.mean(sentence_vector, axis=0)
    return sentence_vector

# apply the generate_word_embedding() function to the preprocessed text data
train_data['embedding'] = train_data['Text'].apply(generate_word_embedding)

# create a new DataFrame for the feature matrix
embedding_size = word2vec_model.vector_size
features_df = pd.DataFrame(train_data['embedding'].tolist(), columns=[f'embedding_{i}' for i in range(embedding_size)])

# perform PCA with n_components set to retain 98% of variance
pca_emb = PCA(n_components=0.98)
features_emb_pca = pca_emb.fit_transform(features_df)

# create a new DataFrame for the PCA features
pca_emb_cols = [f"PC_emb{i+1}" for i in range(features_emb_pca.shape[1])]
pca_df_emb = pd.DataFrame(features_emb_pca, columns=pca_emb_cols)


# create a TF-IDF vectorizer object
tfidf_vectorizer = TfidfVectorizer()

# fit and transform the vectorizer on the preprocessed text data
tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['Text'].apply(lambda x: ' '.join(x)))

tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_features_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names)
## standardize the features
#scaler = StandardScaler()
#features_std = scaler.fit_transform(features)

# perform PCA with n_components set to retain 95% of variance
pca = PCA(n_components=0.95)
features_tfidf_pca = pca.fit_transform(tfidf_features_df)

# create a new DataFrame for the PCA features
pca_tfidf_cols = [f"PC_tfidf{i+1}" for i in range(features_tfidf_pca.shape[1])]
pca_df_tfidf = pd.DataFrame(features_tfidf_pca, columns=pca_tfidf_cols)

# add the TF-IDF features to the feature matrix DataFrame
features_df = pd.concat([pca_df_tfidf, pca_df_emb], axis=1)


# add the number of characters, number of words, and number of capital characters as features
features_df['num_characters'] = train_data['Text'].apply(lambda x: len(' '.join(x)))
features_df['num_words'] = train_data['Text'].apply(lambda x: len(x))

# add the common features from the features.csv file
features_df['num_sentences'] = original_data["Text"].apply(lambda s: s.count('.'))
features_df['num_question_marks'] = original_data["Text"].apply(lambda s: s.count('?'))
features_df['num_exclamation_marks'] = original_data["Text"].apply(lambda s: s.count('!'))
features_df['num_unique_words'] = train_data["Text"].apply(lambda x: len(set(x)))


# add the label column to the feature matrix DataFrame
label = features_df.columns
features_df['Sentiment'] = train_data['Sentiment']

# weight the negative sentiment samples by 1.5
features_df.loc[features_df['Sentiment'] == 'negative',label] *= 2


# save the feature matrix to a CSV file
# pca_df_emb.to_csv("pca_df_emb.csv", index=False)
# pca_df_tfidf.to_csv("pca_df_tfidf.csv", index=False)
features_df.to_csv("features.csv", index=False)
