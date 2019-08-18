# Chapter Three

## Notes

### Stochastic Gradient Decent (SGD)

- Can handle large data sets efficiently because it deals with training instances individually. This makes it very good for online learning
- Relies on randomness during training. It can be configured to have reproducible results

### Performance measures

#### Accuracy

In the example we only trained the classifier to give a true or false prediction based on whether it thought the number was a 5 or not and it achieved a very good result. Over 90%.

This is an example of my accuracy isn't a great performance measure on its own because that is expected if we have a data set of 10 classes.

#### Confusion Matrix

The idea of a confusion matrix is to count the amount of times instances of class A are confused as class B.

A confusion matrix basically give the false positives, true positives, false negatives and true negatives.

#### Precision

This is the accuracy of positive predictions.

Calculated with: TP / (TP + FP)

The issue with just using precision is if you only only evaluate one instance and make sure it is correct. Precision is usually used in conjunction with another performance measure to prevent this behaviour.

#### Recall

This is the ratio of positive instances that are correctly detected by the classifier.

Calculated with: TP / (TP + FN)

#### F1 Score

This is the harmonic mean of precision and recall and is useful as a single measure to assess the performance of a classifier.

The F1 score will only be high if both the precision and recall are high.

Calculated with: 2 * ((P x R) / (P + R))

#### Precision / Recall Tradeoff

Depending on the use case you might not always want both a high precision and a high recall. For example if you were building a classifier to detect if videos are safe for children you would prefer a higher precision, so it keeps safe videos but a lower recall as its best to reject good videos rather than show bad videos.

#### The ROC (Receiver Operating Characteristic) Curve

This is commonly used with binary classifiers. It plots the true positive rate (another name for recall) against the false positive rate. The false positive rate is the ratio of negative instances that are incorrectly classified as positive. It is equal to one minus the true negative rate, which is the ratio of negative instances that are correctly classified as negative. The true negative ratio is also called specificity. The ROC curve plots specificity versus 1 - specificity.

#### Multiclass Classification

A strategy is to train a binary classifier for each class you have. For example if you are trying to classify a digit into a class you would have 10 classes so you would need to train 10 classifiers. When you want to classify a new image you would get a prediction for each model and use the highest score. This is called one-versus-all classification.

A different strategy is to train a classifier for each pair of classes. In the digit example we would have to train 45 classifiers (N x (N - 1) / 2). This is called a one-versus-one strategy. The main advantage of this strategy is that each classifier needs to be trained on a subset of the training data.

There are some classification algorithms which scale poorly so they would be better for a one-versus-one strategy. In most other use cases one-versus-all is preferred.

#### Error Analysis

You can plot a confusion matrix to discover what classes where misclassified.

If you take each error and divide by the total number of images of that class we can compare the error rates rather than the absolute number of errors.

Analysing the confusion matrix can give us insights into how to improve our classifier. We can also gain insights into what the classifier is doing.


