# Chapter Two

## Notes

By using the info function on a Pandas data frame we can see how many rows are missing data for a particular column. This means will we need to take some action to clean up that data before we run our model.

### Correlations

We can calculate the Standard Correlation Coefficient by using the corr function. This will tell us the linear correlation between one feature and every other feature in the data set. This means we can answer questions like "if feature x goes up then feature y generally does up / down. This may miss non-linear relationships. For example answering the question "if feature x is zero then feature y generally goes up".

### Preparing data for machine learning algorithms

Using functions to prepare the data has advantages because:

- We can reuse them on many data sets
- We can build a library for further use
- It can be used in online systems to clean the data before passing it to the algorithm
- We will easily be able to try these transformations easily from here on out

#### Handling missing data

We have three options:

- Get rid of corresponding districts
- Get rid of the whole attribute
- Set values to some value (zero, mean, median)

Scikit learn offers a class to handle dealing with missing data. It's called Imputer.

### Scikit-Learn Design

#### Consistency

All objects share a consistent and simple interface.

- Estimators - An object that can estimate some parameters based on a data set
- Transformers - Transforms a data set
- Predictors - Predictors can make a prediction on a data set. For example the Linear Regression class is a predictor

#### Inspections

All the estimators hyperparameters and learned parameters are accessible directly via public instance variables.

#### Nonproliferation of classes

Data sets are represented as NumPy arrays or SciPy sparse matrices and hyperparameters are regular Python strings or numbers.

#### Composition

Existing building blocks are reused as much as possible.

#### Sensible Defaults

Scikit-learn provides sensible default values for most parameters, making it easy to create a baseline working system quickly.

### Feature Scaling

Generally machine learning algorithms do not perform well when the features have different scales.

Two common approaches to feature scaling is:

- min-max scaling (normalisation)
    - Scales the values to be between 0 and 1
    - x - min / (max - min)
    - Scikit-learn provides the MinMaxScaler
- standardisation
    - (x - mean) / variance
    - Does not bound to a specific range
    - Much less affected by outliers
    - Scikit-learn provides the StandardScaler
    
### Transformation pipelines

We will probably have various transformations that need to be executed in a specific order so we would use a transformation pipeline to achieve this.

Scikit-learn provides a Pipeline class where we can configure a pipeline of transformers.

The pipeline constructor takes a list of name / estimator pairs defining a sequence of steps. It will call the `fit_transform` methods on all the transformers.

### Fine tuning a model

#### Grid Search

Grid search can automatically run experiments to find the optimal hyperparameters for your model.

#### Randomised search

The grid search approach is good for when we are exploring relatively few combinations, but when the hyperparameter space is large it is often better to used a randomised search instead.

Instead of trying out all possible combinations it evaluates a given number of random combinations.

This has two main benefits:
- If you let the randomised search for 1000 iterations it will explore 1000 different values for each hyperparameter
- You have more control over the computing budget you want to allocate to hyperparameter search by simply setting the number of iterations

### Going live

Before a model is launched into a production environment it makes sense to do some things before hand.

- Write code to monitor the systems performance and trigger an alert when it drops
    - Performance degradation
    - Sudden breakage
- A process for human's to analyse the systems performance
- Evaluate the systems input data quality to catch any performance degradation early
- Automatically train models regularly using fresh data