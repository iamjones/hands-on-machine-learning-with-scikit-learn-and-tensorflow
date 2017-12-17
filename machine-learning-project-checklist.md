# Machine Learning Project Checklist

## Frame the problem and look at the big picture

1. Define the objective in business terms.
2. How will your solution be used?
3. What are the current solutions / workarounds? (if any)
4. How should you frame this problem? (supervised / unsupervised, online / offline)
5. How should performance be measured?
6. Is the performance measure aligned with the business objective?
7. What would the minimum performance needed to reach the business objective?
8. What are comparable problems? Can you reuse experience or tools?
9. Is human expertise available?
10. How would you solve the problem manually?
11. List the assumptions you (or others) have made so far.
12. Verify assumptions if possible.

## Get the data

1. List the data you need and how much you need.
2. Find and document where you can get the data.
3. Check how much space it will take.
4. Check legal obligations, and get authorisation if necessary.
5. Get access authorisations.
6. Create a workspace (with enough storage space).
7. Get the data.
8. Convert the data to a format you can easily manipulate (without changing the data itself).
9. Ensure sensitive information is deleted or protected (e.g anonymised).
10. Check the size and type of the data (time series, sample, geographical).
11. Sample a test set, put it aside, and never look at it.

## Explore the data

1. Create a copy of the data for exploration (sampling it down to a manageable size if necessary).
2. Create a Jupyter notebook and keep a record of your exploration.
3. Study each attribute and its characteristics:
    - Name
    - Type (categorical, int / float, bounded / unbounded, text, structured)
    - % of missing values
    - Noisiness and type of noise (stochastic, outliers, rounding errors)
    - Possibility useful for task
    - Type of distribution
4. For supervised learning tasks, identify the target attributes.
5. Visualise the data.
6. Study correlations between attributes.
7. Study how you would solve the problem manually.
8. Identify the promising transformations you may want to apply.
9. Identify extra data that would be useful.
10. Document what you have learned.

## Prepare the data

Notes:
- Work on copies of the data
- Write functions for all data transformations.

1. Data cleaning:
    - Fix or remove outliers.
    - Fill in missing values or drop their rows / columns.

2. Feature selection:
    - Drop the attributes that provide no useful data for the task.

3. Feature engineering:
    - Discretise continuous features.
    - Decompose features (categorical, date / time).
    - Add promising transformations of features.
    - Aggregate features into promising new features.

4. Feature scaling: standardise or normalise features.

## Short-List Promising Models

Notes:
 - If the data is huge, you may want to sample smaller training sets so you can train many different models in a reasonable time.
 - Once again, try to automate these steps as much as possible.

1. Train many quick and dirty models from different categories (linear, naive Bayes, SVM, Random Forests, neural net) using standard parameters.
2. Measure and compare their preforamce:
    - For each model, use N-fold cross validation and compute the mean and standard diviation of the performance measures on the N folds.
3. Analyse the most significant variables for each algorithm.
4. Analyse the types of errors the models make:
    - What data would a human have used to avoid these errors?
5. Have a quick round of feature selection and engineering.
6. Have one or two quick iterations of the fice previous steps.
7. Short-list the top three to five most promising models, preferring models that make different types of errors.

## Fine-Tune the system

Notes:
    - You will want to use as much data as possible for this step, especially as you move toward the end of fine-tuning.
    - Ad always automate what you can.

1. Fine-tune the hyperparameters using cross-validation:
    - Treat your data transformation choices as hyperparameters, expecially when you are not sure about them (should I replace missing values with zero or use the median value? Or should I drop the row?).
    - Unless there are very few hyperparameter values to explore, prefer random search over grid search. If training is very long, you may prefer a Bayesian optimisation approach.
2. Try ensable methods. Combining your best models will often perform better than running them individually.
3. Once you are confident about your final model, measure its performance on the test set to estimate the generalisation error.

## Present the solution

1. Document what you have done.
2. Create a nice presentation:
    - Make sure you highlight the big picture first.
3. Explain why your solution achieves the business objective.
4. Don;t forget to present interesting points you noticed along the way.
    - Describe what working and what did not.
    - List your assumptions and your system's limitations.
5. Ensure your key findings are communicated through beautiful visualisations or easy-to-remember statements.

## Launch!

1. Get your solution ready for production (plug into production data inputs, write tests).
2. Write monitoring code to check your systems live performance at regualar intervals and trigger alerts when it drops.
    - Beware of slow degradation too: models tend to "rot" as data evolves.
    - Measuring performance may require a human pipeline.
    - Also monitor your inputs quality. This is particularly important for online learning systems.
3. Retrain your models ona regular basis on fresh data.
















