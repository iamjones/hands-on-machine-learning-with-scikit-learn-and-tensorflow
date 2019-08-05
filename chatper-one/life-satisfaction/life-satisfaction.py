import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('data.csv', thousands=',')
gdp_data = pd.read_csv('gdp-data.csv', thousands=',', delimiter='\t', encoding='latin1', na_values='n/a')

# Prepare the data


