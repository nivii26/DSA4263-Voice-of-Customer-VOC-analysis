import pandas as pd
import os

from ..data.rawdata_preprocessing import *

# contraction
def test_remove_contraction(in_df, out_df):
	in_text = in_df[in_df["Type"]=="contraction"].Text
	out_text = out_df[out_df["Type"]=="contraction"].Text
	for in_value, out_value in zip(in_text, out_text):
		assert remove_contractions(in_value) == out_value

# emoji
def test_remove_emoji(in_df, out_df):
	in_text = in_df[in_df["Type"]=="emoji"].Text
	out_text = out_df[out_df["Type"]=="emoji"].Text
	for in_value, out_value in zip(in_text, out_text):
		assert remove_emoji(in_value) == out_value

# html
def test_remove_html(in_df, out_df):
	in_text = in_df[in_df["Type"]=="html"].Text
	out_text = out_df[out_df["Type"]=="html"].Text
	for in_value, out_value in zip(in_text, out_text):
		assert remove_html(in_value) == out_value

# digits_word / symbols_word
def test_remove_word_containing_digits_symbols(in_df, out_df):
	in_text = in_df[in_df["Type"]=="digits_word"].Text
	out_text = out_df[out_df["Type"]=="digits_word"].Text
	for in_value, out_value in zip(in_text, out_text):
		assert remove_word_containing_symbols(remove_word_containing_digits(in_value)) == out_value

# digits
def test_remove_digits(in_df, out_df):
	in_text = in_df[in_df["Type"]=="digits"].Text
	out_text = out_df[out_df["Type"]=="digits"].Text
	for in_value, out_value in zip(in_text, out_text):
		assert remove_digits(in_value) == out_value

# extra_spaces
def test_remove_extra_spaces(in_df, out_df):
	in_text = in_df[in_df["Type"]=="extra_spaces"].Text
	out_text = out_df[out_df["Type"]=="extra_spaces"].Text
	for in_value, out_value in zip(in_text, out_text):
		assert remove_extra_spaces(in_value) == out_value

# punctuations
def test_remove_punctuations(in_df, out_df):
	in_text = in_df[in_df["Type"]=="punctuations"].Text
	out_text = out_df[out_df["Type"]=="punctuations"].Text
	for in_value, out_value in zip(in_text, out_text):
		assert remove_punctuations(in_value) == out_value

# rawdata_preprocessing
def test_preprocess_text(in_df, out_df):
	in_text = in_df[in_df["Type"]=="rawdata_preprocessing"].Text
	out_text = out_df[out_df["Type"]=="rawdata_preprocessing"].Text
	for in_value, out_value in zip(in_text, out_text):
		assert preprocess_text(in_value) == out_value

if __name__ == "__main__":
	os.chdir(r"./root/unit_testing")
	input_df = pd.read_csv(r"./testcase/input/test_rawdata_preprocessing_input.csv")
	output_df = pd.read_csv(r"./testcase/output/test_rawdata_preprocessing_output.csv")

	test_remove_contraction(input_df, output_df)
	test_remove_emoji(input_df, output_df)
	test_remove_html(input_df, output_df)
	test_remove_word_containing_digits_symbols(input_df, output_df)
	test_remove_digits(input_df, output_df)
	test_remove_punctuations(input_df, output_df)
	test_remove_extra_spaces(input_df, output_df)
	test_preprocess_text(input_df, output_df)

	print("All Tests Passed")