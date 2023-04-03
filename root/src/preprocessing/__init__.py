"""
Makes the folder a python package in which you can just import their
variables and functions easily

I.E: In other python files, you can do the following:
from data import preprocessed_data # Assuming preprocessed_data variable 
exist
"""
from .tm_preprocessing import tm_preprocess
from .sa_preprocessing import sa_preprocess

