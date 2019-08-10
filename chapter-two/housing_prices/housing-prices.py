from json import encoder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting._matplotlib import scatter_matrix
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

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

print("Linear regression root mean square error: {}".format(lin_rmse))

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)

print("Decision tree root mean square error: {}".format(tree_rmse))

# Decision tree using 10 fold cross validation
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

print("Scores: {}".format(tree_rmse_scores))
print("Mean: {}".format(tree_rmse_scores.mean()))
print("Standard deviation: {}".format(tree_rmse_scores.std()))

# Linear regression using 10 fold cross validation
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

print("Scores: {}".format(lin_rmse_scores))
print("Mean: {}".format(lin_rmse_scores.mean()))
print("Standard deviation: {}".format(lin_rmse_scores.std()))

# Random forest using 10 fold cross validation
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

print("Scores: {}".format(forest_rmse_scores))
print("Mean: {}".format(forest_rmse_scores.mean()))
print("Standard deviation: {}".format(forest_rmse_scores.std()))

# Use Grid Search to find optimal hyperparameters
param_grid = [
    {
        "n_estimators": [3, 10, 30],
        "max_features": [2, 4, 6, 8]
    },
    {
        "bootstrap": [False],
        "n_estimators": [3, 10],
        "max_features": [2, 3, 4]
    }
]

param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 11),
              "min_samples_split": randint(2, 11),
              "bootstrap": [True, False]}

grid_search = RandomizedSearchCV(forest_reg, param_distributions=param_dist, n_iter=20, cv=10, iid=False)
# grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error")
grid_search.fit(housing_prepared, housing_labels)

print("Grid search best params: {}".format(grid_search.best_params_))
print("Grid search best estimator: {}".format(grid_search.best_estimator_))
print("Grid search best score: {}".format(grid_search.best_score_))
print("Grid search best index: {}".format(grid_search.best_index_))

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# Analyse the best models and their errors
feature_importances = grid_search.best_estimator_.feature_importances_
print("Feature importances: {}".format(feature_importances))

# Display the importance scores next to each attribute name
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
# cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs
atts = sorted(zip(feature_importances, attributes), reverse=True)

print("Feature importances with attribute names: {}".format(atts))

# Run the test set against the chosen model
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
Y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("Final prediction: {}".format(final_rmse))
