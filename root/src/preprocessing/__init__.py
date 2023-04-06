"""
Makes the folder a python package in which you can just import their
variables and functions easily

I.E: In other python files, you can do the following:
from data import preprocessed_data # Assuming preprocessed_data variable 
exist
"""
# from .tm_preprocessing import TM_PREPROCESS_TEST
# from .sa_preprocessing import SA_PREPROCESS_TEST, SA_PREPROCESS_TRAIN
# from .rawdata_preprocessing import PREPROCESS_RAW, remove_html
from .tm_preprocessing import *
from .sa_preprocessing import *
from .rawdata_preprocessing import *

