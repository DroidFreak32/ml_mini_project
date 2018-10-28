#!/usr/bin/env python3
import cgitb, cgi
import mysql.connector 
import numpy as np
import pandas as pd
import tkinter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# GUI Initialization
predict_window = tkinter.Tk()
predict_window.title("Python Project")
iris_color = '#43348c'

# GUI - User-entered values
sepal_length = tkinter.DoubleVar()
sepal_width = tkinter.DoubleVar()
petal_length = tkinter.DoubleVar()
petal_width = tkinter.DoubleVar()

# Naive Bayes algorithm
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
# dataset = pd.read_csv('iris_uci.csv')
mydb=mysql.connector.connect(host="localhost",user="root",passwd="root",database="iris_dataset")
mycursor=mydb.cursor()
sql="SELECT * FROM Iris"
dataset=pd.read_sql(sql,mydb)

# Spliting the dataset into independent and dependent variables
X = dataset.iloc[:,1:5].values
y = dataset['species'].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 82)

'''
This function transforms the data such that its distribution will have a mean value 0 and standard deviation of 1.
Here each value in the dataset will have the mean value subtracted, 
and then divided by the standard deviation of the whole dataset.

x′=(x−μ)/σ

fit() calculates μ and σ
fit_transform() calls fit() and transform() internally.

'''
sc = StandardScaler()
X_train = sc.fit_transform(X_train) # Now sc is fitted, so just use sc.transform(test_data) from now on.
X_test = sc.transform(X_test)

# Fitting Naive Bayes Classification to the Training set with linear kernel
nvclassifier = GaussianNB().fit(X_train, y_train)

# Predicting the Test set results. (0 = setosa, 1 = versicolor, 2 = virginica)
predictions = nvclassifier.predict(X_test)
print(X_test)

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
print ("\n\nSide-by-side comparison of Actual vs Predicted values")
print ("-----------------------------------------------------")
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

print('\nCorrect predictions: ', corrPred)
print('False predictions', falsePred)
print ('\n\nAccuracy of the Bayesian Classification is: ', corrPred/(cm.sum())*100, '%')


# GUI - Prediction
def predict():
    print("The entries are ",sepal_length.get()," ",sepal_width.get()," ",petal_length.get()," ",petal_width.get(),"\n\n")
    test = np.array([[sepal_length.get(), sepal_width.get(), petal_length.get(), petal_width.get()]])
    test=sc.transform(test)
    prediction = nvclassifier.predict(test)
    if prediction[0] == 0:
        l_prediction = tkinter.Label(predict_window, text="The flower is Iris-setosa", fg=iris_color)
        l_prediction.grid(row=6)
    elif prediction[0] == 1:
        l_prediction = tkinter.Label(predict_window, text="The flower is Iris-versicolor", fg=iris_color)
        l_prediction.grid(row=6)
    else:
        l_prediction = tkinter.Label(predict_window, text="The flower is Iris-virginica", fg=iris_color)
        l_prediction.grid(row=6)

def prediction_window():

    # GUI - Elements

    l_input_desc = tkinter.Label(predict_window, text="Enter the following parameters of iris flower:")
    l_input_sepal_length = tkinter.Label(predict_window, text="Sepal Length:")
    entry_sepal_length = tkinter.Entry(predict_window, textvariable=sepal_length)

    l_input_sepal_width = tkinter.Label(predict_window, text="Sepal Width:")
    entry_sepal_width = tkinter.Entry(predict_window, textvariable=sepal_width)

    l_input_petal_length = tkinter.Label(predict_window, text="Petal Length:")
    entry_petal_length = tkinter.Entry(predict_window, textvariable=petal_length)

    l_input_petal_width = tkinter.Label(predict_window, text="Petal Width:")
    entry_petal_width = tkinter.Entry(predict_window, textvariable=petal_width)

    button= tkinter.Button(predict_window, text ="Predict!", bd=5, bg="yellow", fg="red", font=20, justify="center", height=3, command=predict)
    l_input_desc.grid(row=0,column=0)

    l_input_sepal_length.grid(row=1,column=0)
    entry_sepal_length.grid(row=1,column=1)
    l_input_sepal_width.grid(row=2,column=0)
    entry_sepal_width.grid(row=2,column=1)
    l_input_petal_length.grid(row=3,column=0)
    entry_petal_length.grid(row=3,column=1)
    l_input_petal_width.grid(row=4,column=0)
    entry_petal_width.grid(row=4,column=1)

    button.grid(row=5,column=0,columnspan=2)

    predict_window.mainloop()

prediction_window()

# def home_page():
# home_page()
