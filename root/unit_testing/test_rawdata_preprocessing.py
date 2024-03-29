import pandas as pd
import os
import pytest

from ..src.preprocessing.rawdata_preprocessing import *

@pytest.fixture
def generate_data():
	in_df = pd.read_csv(r"./root/unit_testing/testcase/input/test_rawdata_preprocessing_input.csv")
	out_df = pd.read_csv(r"./root/unit_testing/testcase/output/test_rawdata_preprocessing_output.csv")
	return in_df, out_df

# contraction
def test_remove_contraction(generate_data):
	in_df, out_df = generate_data
	in_text = in_df[in_df["Type"]=="contraction"].Text
	out_text = out_df[out_df["Type"]=="contraction"].Text
	for in_value, out_value in zip(in_text, out_text):
		assert remove_contractions(in_value) == out_value

# emoji
def test_remove_emoji(generate_data):
	in_df, out_df = generate_data
	in_text = in_df[in_df["Type"]=="emoji"].Text
	out_text = out_df[out_df["Type"]=="emoji"].Text
	for in_value, out_value in zip(in_text, out_text):
		assert remove_emoji(in_value) == out_value

# html
def test_remove_html(generate_data):
	in_df, out_df = generate_data
	in_text = in_df[in_df["Type"]=="html"].Text
	out_text = out_df[out_df["Type"]=="html"].Text
	for in_value, out_value in zip(in_text, out_text):
		assert remove_html(in_value) == out_value

# digits_word / symbols_word
def test_remove_word_containing_digits_symbols(generate_data):
	in_df, out_df = generate_data
	in_text = in_df[in_df["Type"]=="digits_word"].Text
	out_text = out_df[out_df["Type"]=="digits_word"].Text
	for in_value, out_value in zip(in_text, out_text):
		assert remove_word_containing_symbols(remove_word_containing_digits(in_value)) == out_value

# digits
def test_remove_digits(generate_data):
	in_df, out_df = generate_data
	in_text = in_df[in_df["Type"]=="digits"].Text
	out_text = out_df[out_df["Type"]=="digits"].Text
	for in_value, out_value in zip(in_text, out_text):
		assert remove_digits(in_value) == out_value

# extra_spaces
def test_remove_extra_spaces(generate_data):
	in_df, out_df = generate_data
	in_text = in_df[in_df["Type"]=="extra_spaces"].Text
	out_text = out_df[out_df["Type"]=="extra_spaces"].Text
	for in_value, out_value in zip(in_text, out_text):
		assert remove_extra_spaces(in_value) == out_value

# punctuations
def test_remove_punctuations(generate_data):
	in_df, out_df = generate_data
	in_text = in_df[in_df["Type"]=="punctuations"].Text
	out_text = out_df[out_df["Type"]=="punctuations"].Text
	for in_value, out_value in zip(in_text, out_text):
		assert remove_punctuations(in_value) == out_value

# rawdata_preprocessing
def test_preprocess_text(generate_data):
	in_df, out_df = generate_data
	in_text = in_df[in_df["Type"]=="rawdata_preprocessing"].Text
	out_text = out_df[out_df["Type"]=="rawdata_preprocessing"].Text
	for in_value, out_value in zip(in_text, out_text):
		assert preprocess_text(in_value) == out_value

# if __name__ == "__main__":
# 	os.chdir(r"./root/unit_testing")
# 	input_df = pd.read_csv(r"./testcase/input/test_rawdata_preprocessing_input.csv")
# 	output_df = pd.read_csv(r"./testcase/output/test_rawdata_preprocessing_output.csv")

#	generate_data = (input_df, output_df)

# 	test_remove_contraction(generate_data)
# 	test_remove_emoji(generate_data)
# 	test_remove_html(generate_data)
# 	test_remove_word_containing_digits_symbols(generate_data)
# 	test_remove_digits(generate_data)
# 	test_remove_punctuations(generate_data)
# 	test_remove_extra_spaces(generate_data)
# 	test_preprocess_text(generate_data)

# 	print("All Tests Passed")