import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class GaussianNB(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        """Fit Gaussian Naive Bayes according to X, y

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values."""

        separated = [
            [
                x for x,
                t in zip(X, y)
                if t == c
            ]
            for c in np.unique(y)
        ]
        self.model = np.array(
            [
                np.c_[
                    np.mean(i, axis=0),
                    np.std(i, axis=0)
                ]
                    for i in separated
            ]
        )
        return self

    def _prob(self, x, mean, std):
        exponent = np.exp(- ((x - mean)**2 / (2 * std**2)))
        return np.log(exponent / (np.sqrt(2 * np.pi) * std))

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]"""
        return [
            [
                sum(
                    self._prob(i, *s) for s,
                    i in zip(summaries, x)
                )
                for summaries in self.model
            ] 
            for x in X
        ]

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]"""
        return np.argmax(self.predict_log_proba(X), axis=1)

# Importing the dataset
dataset = pd.read_csv('iris_uci.csv')

# Spliting the dataset into independent and dependent variables
X = dataset.iloc[:,:4].values
y = dataset['species'].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 82)

# Feature Scaling to bring the variable in a single scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes Classification to the Training set with linear kernel
nvclassifier = GaussianNB().fit(X_train, y_train)

# Predicting the Test set results. (0 = setosa, 1 = versicolor, 2 = virginica)
predictions = nvclassifier.predict(X_test)

# Assigning flower class based on the prediction results.
y_pred = []
for x in range(len(predictions)):
    if predictions[x] == 0:
        y_pred.append('Iris-setosa')
    elif predictions[x] == 1:
        y_pred.append('Iris-versicolor')
    else:
        y_pred.append('Iris-virginica')

y_pred = np.asarray(y_pred)
print("Actual values: \n",y_test)
print("Predicted values: \n",y_pred)

# Side-by-side view of actual and predicted values
y_compare = np.vstack((y_test,y_pred)).T

# Actual value on the left side and predicted value on the right hand side
print(y_compare[:,:])

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Finding accuracy from the confusion matrix.
a = cm.shape
corrPred = 0
falsePred = 0

for row in range(a[0]):
    for c in range(a[1]):
        if row == c:
            corrPred +=cm[row,c]
        else:
            falsePred += cm[row,c]

print('Correct predictions: ', corrPred)
print('False predictions', falsePred)
print ('\n\nAccuracy of the Bayesian Classification is: ', corrPred/(cm.sum())*100, '%')
