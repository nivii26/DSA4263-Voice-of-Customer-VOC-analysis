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

def preprocess_text_flair(reviewText):
	"""
	Cleans Text Data
	Input: 
	reviewText (String)
	Output:
	reviewText (String)
	"""
	# Remove html
	reviewText = re.sub(r"<[^>]+>", " ", reviewText)
	return reviewText

def preprocess_text(reviewText):
	"""
	Cleans Text Data
	Input: 
	reviewText (String)
	Output:
	reviewText (String)
	"""
	# Change contractions to words
	reviewText = " ".join(contractions.fix(word) for word in reviewText.split())
	# Remove emojis
	reviewText = demoji.replace(reviewText, "")
	# Remove html
	reviewText = re.sub(r"<[^>]+>", " ", reviewText)
	# To Lower Case
	reviewText = reviewText.lower()
	# Words containing digits
	reviewText = re.sub('\w*\d\w*','', reviewText)
	# Remove digits
	reviewText = re.sub("[^a-zA-Z]+", " ", reviewText)
	# Remove Extra Spaces
	reviewText = re.sub(' +',' ', reviewText)
	# Remove punctuations
	reviewText = re.sub('[%s]' % re.escape(string.punctuation), '', reviewText)
	return reviewText

if __name__ == "__main__":

	"""
	1. Generate a Basic EDA (Missing, Duplicate, Distribution) report for new Raw Data which can
	be found under data/processed/reports folder
	2. Clean ALL csv datasets in the Raw Folder and combine them into 1 cleaned dataframe which is 
	saved in processed folder with the filename: [datetime of run]_CLEANED_DATA.csv
	Input: 
	All Raw CSV files in the data/raw directory [NOTE: Columns have to be "Sentiment", "Time", "Text"]
	#TODO: Streamline Code to Check for the format above 
	Output:
	Combined cleaned dataset with the filename [datetime of run]_CLEANED_DATA.csv suitable for all NLP tasks
	"""
	
	current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	final_cleaned_data = pd.DataFrame(columns=["Sentiment", "Time", "Text"])
	final_cleaned_data_flair = pd.DataFrame(columns=["Sentiment", "Time", "Text"])
	
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
			cleaned_data_flair = raw_data.dropna().drop_duplicates()
			## Preprocess the Review Column
			cleaned_data["Text"] = cleaned_data["Text"].apply(preprocess_text)
			cleaned_data_flair["Text"] = cleaned_data["Text"].apply(preprocess_text_flair)

			## Combine all the cleaned datasets
			final_cleaned_data = pd.concat([final_cleaned_data, cleaned_data])
			final_cleaned_data_flair = pd.concat([final_cleaned_data_flair, cleaned_data_flair])

	final_cleaned_data.to_csv(f"{current_time}_CLEANED_DATA.csv", index = False)
	final_cleaned_data_flair.to_csv(f"../../src/data/sa/CLEANED_DATA_flair.csv", index = False)
