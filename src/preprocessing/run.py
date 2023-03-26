from feature_engineering_sa import *
from sklearn.model_selection import train_test_split
# load the data from the csv file
reviews = pd.read_csv("reviews.csv")
# split data by Sentiment
positive_reviews = reviews[reviews.Sentiment == 'positive']
negative_reviews = reviews[reviews.Sentiment == 'negative']

# split data by 3:7
positive_train, positive_test = train_test_split(positive_reviews, test_size=0.3)
negative_train, negative_test = train_test_split(negative_reviews, test_size=0.3)

# merge data
train_data = pd.concat([positive_train, negative_train])
test_data = pd.concat([positive_test, negative_test])

train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)

train_feature,word2vec_model,tfidf, pca_emb, pca_tfidf = features_sa_train(train_data)
train_feature.to_csv("features_train.csv", index=False)

test_feature, sentiment = features_sa_test(test_data,word2vec_model,tfidf, pca_emb, pca_tfidf)
print(test_feature["Sentiment"])
test_feature.to_csv("features_test.csv", index=False)
