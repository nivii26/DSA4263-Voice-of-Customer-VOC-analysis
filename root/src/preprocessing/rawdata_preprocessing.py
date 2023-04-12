import os
import re
import datetime
import string
import pandas as pd
from ydata_profiling import ProfileReport

# Other NLP libraries
import contractions
import demoji

def remove_contractions(text):
	return " ".join(contractions.fix(word) for word in text.split())

def remove_emoji(text):
	return demoji.replace(text, "")

def remove_html(text):
	return re.sub(r"<[^>]+>", "", text)

def remove_word_containing_digits(text):
	return re.sub('\w*\d\w*','', text)

def remove_word_containing_symbols(text):
	return re.sub(r"\b\w+[^\s\w]\w+\b", '', text)

def remove_digits(text):
	return re.sub("\d+", "", text)

def remove_punctuations(text):
	return re.sub('[%s]' % re.escape(string.punctuation), '', text)

def remove_extra_spaces(text):
	return re.sub(' +',' ', text)

def preprocess_text(reviewText):
	"""
	Cleans Text Data

	Input: 
	reviewText (String)

	Output:
	reviewText (String)
	"""
	# Change contractions to words
	reviewText = remove_contractions(reviewText)
	# Remove emojis
	reviewText = remove_emoji(reviewText) # Emoji does not work in CSV, only Excel -> Automatically replaced with ??
	# Remove html
	reviewText = remove_html(reviewText)
	# Words containing digits
	reviewText = remove_word_containing_digits(reviewText)
	# Remove digits
	reviewText = remove_digits(reviewText)
	# To Lower Case
	reviewText = reviewText.lower()
	# Words containing Symbols
	reviewText = remove_word_containing_symbols(reviewText)
	# Remove punctuations
	reviewText = remove_punctuations(reviewText)
	# Remove Extra Spaces
	reviewText = remove_extra_spaces(reviewText)
	return reviewText

def PREPROCESS_RAW(RAW_DF, CURRENT_TIME="", SAVE=True):
	## Clean the data
	CLEANED_DF = RAW_DF.dropna().drop_duplicates()
	## Preprocess the Review Column
	CLEANED_DF["Text"] = CLEANED_DF["Text"].apply(preprocess_text)
	#if SAVE:
	#	ProfileReport(RAW_DF).to_file(rf'./root/data/processed/report/{CURRENT_TIME}_DATA_REPORT.html')
	#	CLEANED_DF.to_csv(rf"./root/data/processed/{CURRENT_TIME}_CLEANED_DF.csv")
	return CLEANED_DF

# FOR RUNNING LOCALLY
if __name__ == "__main__":

	"""
	1. Generate a Basic EDA (Missing, Duplicate, Distribution) report for new Raw Data which can
	be found under data/processed/report folder
	2. Clean ALL csv datasets in the Raw Folder and combine them into 1 cleaned dataframe which is 
	saved in processed folder with the filename: [datetime of run]_CLEANED_DATA.csv

	Input: 
	All Raw CSV files in the data/raw directory [NOTE: Columns have to be "Sentiment", "Time", "Text"]

	Output:
	Combined cleaned dataset with the filename [datetime of run]_CLEANED_DATA.csv suitable for all NLP tasks
	"""
	os.chdir(r"./root/src/preprocessing")
	current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	final_cleaned_data = pd.DataFrame(columns=["Sentiment", "Time", "Text"])
	
	for file in os.listdir(r"../../data/raw"):
		if file.endswith(".csv"):
			# Loading of Data
			raw_data = pd.read_csv(rf"../../data/raw/{file}")
			
			# Basic Exploratory Data Analysis (EDA)
			## Generate a report on the data (missing, duplicates)
			if "report" not in os.listdir(r"../../data/processed"):
				os.makedirs(r"../../data/processed/report")
			ProfileReport(raw_data).to_file(rf'../../data/processed/report/{current_time}_DATA_REPORT.html')

			cleaned_data = PREPROCESS_RAW(raw_data, current_time, False)

			## Combine all the cleaned datasets
			final_cleaned_data = pd.concat([final_cleaned_data, cleaned_data])

	final_cleaned_data.to_csv(fr"../../data/processed/{current_time}_CLEANED_DF.csv", index = False)
