import os
import re
import datetime
import string
import pandas as pd
from ydata_profiling import ProfileReport

# Gensim
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# Other NLP libraries
import contractions
import demoji


def preprocess_text(reviewText):
	# Remove html
	reviewText = re.sub(r"<[^>]+>", " ", reviewText)
	# To Lower Case
	reviewText = reviewText.lower()
	# Remove digits
	reviewText = re.sub("[^a-zA-Z]+", " ", reviewText)
	# Words containing digits
	reviewText = re.sub('\w*\d\w*','', reviewText)
	# Remove punctuations
	reviewText = re.sub('[%s]' % re.escape(string.punctuation), '', reviewText)
	# Remove Extra Spaces
	reviewText = re.sub(' +',' ', reviewText)
	# Remove emojis
	reviewText = demoji.replace(reviewText, "")
	# Change contractions to words
	reviewText = " ".join(contractions.fix(word) for word in reviewText.split())
	return reviewText

if __name__ == "__main__":
	
	current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	positive_data = pd.DataFrame(columns=["Sentiment", "Time", "Text"])
	negative_data = pd.DataFrame(columns=["Sentiment", "Time", "Text"])
	
	for file in os.listdir(r"../raw"):
		if file.endswith(".csv"):
			# Loading of Data
			raw_data = pd.read_csv(rf"../raw/{file}")

			# Basic Exploratory Data Analysis (EDA)
			## Generate a report on the data (missing, duplicates)
			if "reports" not in os.listdir():
				os.makedirs(r"./reports")
			ProfileReport(raw_data).to_file(rf'./reports/{current_time}_{file[:-4]}_report.html')

			## Clean the data
			cleaned_data = raw_data.dropna().drop_duplicates()
			## Preprocess the Review Column
			cleaned_data["Text"] = cleaned_data["Text"].apply(preprocess_text)

			## Separate into sentiments for topic modelling
			positive_data = pd.concat([positive_data, cleaned_data.loc[cleaned_data["Sentiment"] == "positive"]])
			negative_data = pd.concat([negative_data, cleaned_data.loc[cleaned_data["Sentiment"] == "negative"]])

	positive_data.to_csv(f"{current_time}_TM_POS_DATA.csv", index = False)
	negative_data.to_csv(f"{current_time}_TM_NEG_DATA.csv", index = False)