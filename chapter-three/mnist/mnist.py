from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, \
    precision_recall_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict
import matplotlib.pyplot as plt


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")


mnist = fetch_openml("mnist_784", version=1, cache=True)
X = mnist["data"]
y = mnist["target"]

some_digit = X[36000]

# Create our training sets and test sets
X_train = X[:60000]
y_train = y[:60000]

X_test = X[60000:]
y_test = y[60000:]

# Shuffle the training set to make sure our cross validation folds are similar
shuffle_index = np.random.permutation(60000)

X_train = X_train[shuffle_index]
y_train = y_train[shuffle_index]

# Start with only classifying one digit. 5 in this case
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)

cross_val = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print("Cross validation scores: {}".format(cross_val))

# Set a set of predictions so we can use ti to form a confusion matrix
# We can use K-fold cross prediction to get the predictions for each fold
y_train_predict = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
cm = confusion_matrix(y_train, y_train_predict)

print("Confusion Matrix: {}".format(cm))

# Calculate the precision
precision = precision_score(y_train, y_train_predict)
print("Precision: {}".format(precision))

# Calculate the recall
recall = recall_score(y_train, y_train_predict)
print("Recall: {}".format(recall))

# Calculate F1 score
f1 = f1_score(y_train, y_train_predict)
print("F1 Score: {}".format(f1))

y_scores = sgd_clf.decision_function([some_digit])
print("Scores: {}".format(y_scores))
threshold = 0
y_scores = cross_val_predict(sgd_clf, X_train, y_train, cv=3, method="decision_function")
print("Cross validation predict: {}".format(y_scores))
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
print("Precisions: {}".format(precisions))
print("Recalls: {}".format(recalls))
print("Thresholds: {}".format(thresholds))

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# Plot a ROC curve
fpr, tpr, thresholds = roc_curve(y_train, y_scores)
plot_roc_curve(fpr, tpr)
plt.show()

roc_score = roc_auc_score(y_train, y_scores)
print("ROC AUC score: {}".format(roc_score))

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, trp_forest, thresholds_forest = roc_curve(y_train, y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, trp_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

forest_roc_score = roc_auc_score(y_train, y_scores_forest)
print("Forest ROC AUC score: {}".format(forest_roc_score))
