import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting._matplotlib import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from housing_prices.combined_attributes_adder import CombinedAttributesAdder


# View the distribution of the data
from housing_prices.data_frame_selector import DataFrameSelector


def plot_hist(the_data):
    the_data.hist(bins=50, figsize=(20, 15))
    plt.show()


# Extract 20% of the data for our test set ans 80% for our training set
def split_train_test(the_data, test_size):
    shuffled = np.random.permutation(len(the_data))
    test_set_size = int(len(the_data) * test_size)
    test_indices = shuffled[:test_set_size]
    training_indices = shuffled[test_set_size:]
    return the_data.iloc[training_indices], the_data.iloc[test_indices]


# Plot a scatter that high lights population density and median house prices
def plot_density_and_median_housing_value():
    data_training.plot(
        kind='scatter', x='longitude', y='latitude', alpha=0.1,
        s=data_training['population'] / 100, label='population', figsize=(10, 7),
        c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True
    )
    plt.legend()
    plt.show()


# Check for correlation with the scatter matrix function
def plot_standard_correlation_coefficient():
    attr = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    scatter_matrix(data_training[attr], figsize=(12, 8))
    plt.show()


# Show all columns when printing stuff
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

housing = pd.read_csv('data.csv')

# We will need to make sure we use stratified sampling to ensure our test set represents the overall data set accurately
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# We should now remove the income_category column from the data to put it back to it original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

# Copy the training set so we can look at it without harming the actual training set
housing = strat_train_set.copy()

# Find any correlations between the features
corr_matrix = housing.corr()

# Extract the housing labels
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

# Get the numerical attributes only
housing_num = housing.drop('ocean_proximity', axis=1)

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('one_hot_encoder', OneHotEncoder(sparse=False))
])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])

housing_prepared = full_pipeline.fit_transform(housing)

# Train a linear regression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

print(lin_rmse)
