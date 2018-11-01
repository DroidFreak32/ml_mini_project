#!/usr/bin/env python3
import mysql.connector
import numpy as np
import pandas as pd
from tkinter import *
from PIL import ImageTk, Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

global predict_window

# GUI Initialization
home_screen = Tk()
home_screen.title("Python Project")
iris_color = '#43348c'

# GUI - User-entered values
sepal_length = DoubleVar()
sepal_width = DoubleVar()
petal_length = DoubleVar()
petal_width = DoubleVar()

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
mydb = mysql.connector.connect(
    host="localhost", user="root", passwd="root", database="iris_dataset")
mycursor = mydb.cursor()
sql = "SELECT * FROM Iris"
dataset = pd.read_sql(sql, mydb)

# Spliting the dataset into independent and dependent variables
X = dataset.iloc[:, 1:5].values
y = dataset['species'].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=82)

'''
This function transforms the data such that its distribution will have a mean value 0 and standard deviation of 1.
Here each value in the dataset will have the mean value subtracted, 
and then divided by the standard deviation of the whole dataset.

x′=(x−μ)/σ

fit() calculates μ and σ
fit_transform() calls fit() and transform() internally.

'''
sc = StandardScaler()
# Now sc is fitted, so just use sc.transform(test_data) from now on.
X_train = sc.fit_transform(X_train)
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
print("Actual values: \n", y_test)
print("Predicted values: \n", y_pred)

# Side-by-side view of actual and predicted values
y_compare = np.vstack((y_test, y_pred)).T

# Actual value on the left side and predicted value on the right hand side
print("\n\nSide-by-side comparison of Actual vs Predicted values")
print("-----------------------------------------------------")
print(y_compare[:, :])

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Finding accuracy from the confusion matrix.
a = cm.shape
corrPred = 0
falsePred = 0

for row in range(a[0]):
    for c in range(a[1]):
        if row == c:
            corrPred += cm[row, c]
        else:
            falsePred += cm[row, c]

print('\nCorrect predictions: ', corrPred)
print('False predictions', falsePred)
print('\n\nAccuracy of the Bayesian Classification is: ',
      corrPred / (cm.sum()) * 100, '%')

# GUI - Show flower window
def show_flower(flower):
    flower_window = Toplevel(home_screen)
    flower_window.title("Predicted Flower")
    flower_canvas = Canvas(flower_window, width=300, height=300)
    flower_canvas.pack()
    if flower == 0:
        img = ImageTk.PhotoImage(Image.open("iris_setosa.jpg"))
        l_flower = Label(flower_window, text="The flower is Iris - Setosa")
        l_flower.config(font=("Ubuntu Mono", 13))
    elif flower == 1:
        img = ImageTk.PhotoImage(Image.open("iris_versicolor.jpg"))
        l_flower = Label(flower_window, text="The flower is Iris - Versicolor")
        l_flower.config(font=("Ubuntu Mono", 13))
    else:
        img = ImageTk.PhotoImage(Image.open("iris_virginica.jpg"))
        l_flower = Label(flower_window, text="The flower is Iris - Virginica")
        l_flower.config(font=("Ubuntu Mono", 13))
    l_flower.pack()
    flower_canvas.create_image(0, 0, anchor=NW, image=img)
    flower_window.mainloop()

# GUI - Prediction
def predict():

    print("The entries are ", sepal_length.get(), " ", sepal_width.get(),
          " ", petal_length.get(), " ", petal_width.get(), "\n\n")
    test = np.array([[sepal_length.get(), sepal_width.get(),
                      petal_length.get(), petal_width.get()]])
    test = sc.transform(test)
    prediction = nvclassifier.predict(test)
    if prediction[0] == 0:
        show_flower(0)
    elif prediction[0] == 1:
        show_flower(1)
    else:
        show_flower(2)


def prediction_window():
    predict_window = Toplevel(home_screen)
    predict_window.title("Predict Flower")

    l_input_desc = Label(
        predict_window, text="Enter the following parameters of iris flower:", font=("Ubuntu Mono", 13))
    l_input_sepal_length = Label(predict_window, text="Sepal Length:")
    l_input_sepal_length.config(font=("Ubuntu Mono", 13))
    entry_sepal_length = Entry(
        predict_window, textvariable=sepal_length)

    l_input_sepal_width = Label(predict_window, text="Sepal Width:")
    l_input_sepal_width.config(font=("Ubuntu Mono", 13))
    entry_sepal_width = Entry(predict_window, textvariable=sepal_width)

    l_input_petal_length = Label(predict_window, text="Petal Length:")
    l_input_petal_length.config(font=("Ubuntu Mono", 13))
    entry_petal_length = Entry(
        predict_window, textvariable=petal_length)

    l_input_petal_width = Label(predict_window, text="Petal Width:")
    l_input_petal_width.config(font=("Ubuntu Mono", 13))
    entry_petal_width = Entry(predict_window, textvariable=petal_width)

    predict_button = Button(predict_window, text="Predict!",
                            bd=1, font=20, justify="center", height=1, command=predict)
    l_input_desc.grid(row=0, column=0)

    l_input_sepal_length.grid(row=1, column=0)
    entry_sepal_length.grid(row=1, column=1)
    l_input_sepal_width.grid(row=2, column=0)
    entry_sepal_width.grid(row=2, column=1)
    l_input_petal_length.grid(row=3, column=0)
    entry_petal_length.grid(row=3, column=1)
    l_input_petal_width.grid(row=4, column=0)
    entry_petal_width.grid(row=4, column=1)

    predict_button.grid(row=5, column=0, columnspan=2)

    predict_window.mainloop()


def home_page():
    title = Label(
        home_screen, text="Hello! Welcome to Iris flower predictor!", height=5)
    title.config(font=("Ubuntu Mono", 16))
    about_iris_button = Button(home_screen, text="About Iris flowers", bd=1, font=20, justify="center", height=1,
                               command=about_iris_flower)
    about_iris_button.config(font=("Ubuntu Mono", 13))
    predict_window_button = Button(home_screen, text="Predict the flowers!", bd=1, font=20, justify="center", height=1,
                                   command=prediction_window)
    predict_window_button.config(font=("Ubuntu Mono", 13))
    about_team_button = Button(home_screen, text="About Team", bd=1, font=20, justify="center", height=1,
                               command=about_team)
    about_team_button.config(font=("Ubuntu Mono", 13))

    canvas = Canvas(home_screen, width=300, height=300)
    canvas.grid(row=1, column=0)
    img = ImageTk.PhotoImage(Image.open("about_iris.jpg"))
    canvas.create_image(0, 0, anchor=NW, image=img)
    canvas2 = Canvas(home_screen, width=300, height=300)
    canvas2.grid(row=1, column=1)
    img2 = ImageTk.PhotoImage(Image.open("predict.jpg"))
    canvas2.create_image(0, 0, anchor=NW, image=img2)
    canvas3 = Canvas(home_screen, width=300, height=300)
    canvas3.grid(row=1, column=2)
    img3 = ImageTk.PhotoImage(Image.open("about_team.jpg"))
    canvas3.create_image(0, 0, anchor=NW, image=img3)

    title.grid(row=0, columnspan=3)
    about_iris_button.grid(row=2)
    predict_window_button.grid(row=2, column=1)
    about_team_button.grid(row=2, column=2)
    home_screen.mainloop()


def about_iris_flower():
    about_iris_window = Toplevel(home_screen)
    about_iris_window.title("About Iris Flowers")

    iris_desc = "Iris is a genus of 260–300 species of flowering plants with showy flowers.\n\nIt takes its name from the Greek word for a rainbow, which is also the name\nfor the Greek goddess of the rainbow, Iris.\n\nSome authors state that the name refers to the wide variety of flower colors\nfound among the many species.\n\nAs well as being the scientific name, iris is also widely used as a common name for all Iris species,\nas well as some belonging to other closely related genera.\n"
    iris_desc += "\nWe will be looking at the following three classes of Iris flower:\n"
    iris_desc += "------------------------------------------------------------------\n"
    title = Label(about_iris_window, text=iris_desc,
                  justify="left", font=("Ubuntu Mono", 13))
    title.grid(row=0, columnspan=3)

    setosa_head = "1) Iris Setosa:\n"
    setosa_head += "--------------\n"
    l_setosa = Label(about_iris_window, text=setosa_head,
                     anchor=W, justify="left", font=("Ubuntu Mono", 13))
    l_setosa.grid(row=2)

    canvas = Canvas(about_iris_window, width=300, height=300)
    canvas.grid(row=3, column=0)
    img = ImageTk.PhotoImage(Image.open("iris_setosa.jpg"))
    canvas.create_image(0, 0, anchor=NW, image=img)

    versicolor_head = "2) Iris Versicolor:\n"
    versicolor_head += "--------------\n"
    l_setosa = Label(about_iris_window, text=versicolor_head,
                     anchor=W, justify="left", font=("Ubuntu Mono", 13))
    l_setosa.grid(row=2, column=1)

    canvas2 = Canvas(about_iris_window, width=300, height=300)
    canvas2.grid(row=3, column=1)
    img2 = ImageTk.PhotoImage(Image.open("iris_versicolor.jpg"))
    canvas2.create_image(0, 0, anchor=NW, image=img2)

    virginica_head = "2) Iris Virginica:\n"
    virginica_head += "--------------\n"
    l_setosa = Label(about_iris_window, text=virginica_head,
                     anchor=W, justify="left", font=("Ubuntu Mono", 13))
    l_setosa.grid(row=2, column=2)

    canvas3 = Canvas(about_iris_window, width=300, height=300)
    canvas3.grid(row=3, column=2)
    img3 = ImageTk.PhotoImage(Image.open("iris_virginica.jpg"))
    canvas3.create_image(0, 0, anchor=NW, image=img3)

    about_iris_window.mainloop()


def about_team():
    about_team_window = Toplevel(home_screen)
    about_team_window.title("About the Team")

    team_name_1 = "Safa Suleman Shaikh"
    team_USN_1 = "USN:           4NM15CS144"

    team_name_2 = "Rushab Shah"
    team_USN_2 = "USN:           4NM15CS141"

    team_semester = "Semester:      7th"
    team_section = "Section:       C"
    team_dept = "Department:    Computer Science and Engineering"

    l_name_1 = Label(about_team_window, text=team_name_1,
                     anchor='w', width=50, font=("Ubuntu Mono", 15))
    l_usn_1 = Label(about_team_window, text=team_USN_1,
                    anchor='w', width=50, font=("Ubuntu Mono", 13))
    l_semester_1 = Label(about_team_window, text=team_semester,
                         anchor='w', width=50, font=("Ubuntu Mono", 13))
    l_section_1 = Label(about_team_window, text=team_section,
                        anchor='w', width=50, font=("Ubuntu Mono", 13))
    l_dept_1 = Label(about_team_window, text=team_dept,
                     anchor='w', width=50, font=("Ubuntu Mono", 13))

    l_name_2 = Label(about_team_window, text="\n"+team_name_2,
                     anchor='w', width=50, font=("Ubuntu Mono", 15))
    l_usn_2 = Label(about_team_window, text=team_USN_2,
                    anchor='w', width=50, font=("Ubuntu Mono", 13))
    l_semester_2 = Label(about_team_window, text=team_semester,
                         anchor='w', width=50, font=("Ubuntu Mono", 13))
    l_section_2 = Label(about_team_window, text=team_section,
                        anchor='w', width=50, font=("Ubuntu Mono", 13))
    l_dept_2 = Label(about_team_window, text=team_dept,
                     anchor='w', width=50, font=("Ubuntu Mono", 13))

    l_name_1.grid(row=0, columnspan=2)
    l_usn_1.grid(row=1, columnspan=2)
    l_semester_1.grid(row=2, columnspan=2)
    l_section_1.grid(row=3, columnspan=2)
    l_dept_1.grid(row=4, columnspan=2)
    l_name_2.grid(row=6, columnspan=2)
    l_usn_2.grid(row=7, columnspan=2)
    l_semester_2.grid(row=8, columnspan=2)
    l_section_2.grid(row=9, columnspan=2)
    l_dept_2.grid(row=10, columnspan=2)

    about_team_window.mainloop()


home_page()
