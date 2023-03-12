import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

# download necessary NLTK data (only need to run this once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# load the data from the csv file
train_data = pd.read_csv("reviews.csv")
# train_label = train_data["Sentiment"]
# train_data = train_data["Text"]

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

# apply the preprocessing function to the text data
train_data['Text'] = train_data['Text'].apply(preprocess_text)

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

# add the label column to the feature matrix DataFrame
features_df['Sentiment'] = train_data['Sentiment']

# create a TF-IDF vectorizer object
tfidf_vectorizer = TfidfVectorizer()

# fit and transform the vectorizer on the preprocessed text data
tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['Text'].apply(lambda x: ' '.join(x)))

# add the TF-IDF features to the feature matrix DataFrame
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_features_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names)

features_df = pd.concat([tfidf_features_df, features_df ], axis=1)

# add the number of characters, number of words, and number of capital characters as features
features_df['num_characters'] = train_data['Text'].apply(lambda x: len(' '.join(x)))
features_df['num_words'] = train_data['Text'].apply(lambda x: len(x))
features_df['num_capital_chars'] = train_data['Text'].apply(lambda x: sum(1 for c in ' '.join(x) if c.isupper()))

# add the common features from the features.csv file
features_df['num_sentences'] = train_data['Text'].apply(lambda x: len(re.findall(r'\.', ' '.join(x))))
features_df['num_question_marks'] = train_data['Text'].apply(lambda x: len(re.findall(r'\?', ' '.join(x))))
features_df['num_exclamation_marks'] = train_data['Text'].apply(lambda x: len(re.findall(r'!', ' '.join(x))))
features_df['num_unique_words'] = train_data['Text'].apply(lambda x: len(set(x)))


# save the feature matrix to a CSV file
features_df.to_csv("features.csv", index=False)
