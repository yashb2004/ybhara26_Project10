"""
predict.py

Yash Bharadwaj
Spring 2023
CS152 

This program trains and tests ML classifiers which are then used to for cancer prognosis based on user input data values through a GUI

To run this program type the following on the commandline:
python3 predict.py

"""
import tkinter as tk
from tkinter import ttk
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Splash(tk.Toplevel):


    def __init__(self, parent):
        """Initializes the Splash object to create a splash screen"""
        super().__init__(parent)
        self.title("Welcome")
        self.geometry("1000x500")
        self.resizable(False, False)
        
        self.label = ttk.Label(self, text='Welcome to the Cancer Diagnosis Classifier!\nIn the other window, please select a ML model and enter the following data values in this order:\n"radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean",\n"concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se",\n"compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst",\n"area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"')
        self.label.pack(padx=50, pady=50)
        
        self.button = ttk.Button(self, text="Close", command=self.destroy)
        self.button.pack(pady=20)


class App(tk.Tk):
    

    def __init__(self):
        """Initializes the App object and assigns it numerous attributes useful for the GUI"""
        super().__init__()
        self.title("Cancer Diagnosis Classifier")
        self.geometry("500x350")
        self.resizable(False, False)
        
        self.model_options = ['Logistic Regression', 'K-NN', 'SVM', 'K-SVM', 'Naive Bayes', 'Decision Tree', 'Random Forest']
        
        self.model_label = ttk.Label(self, text="Select a model:")
        self.model_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        
        self.model_var = tk.StringVar(self)
        self.model_dropdown = ttk.Combobox(self, values=self.model_options, state="readonly", textvariable=self.model_var)
        self.model_dropdown.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)
        self.model_dropdown.current(0)
        
        self.accuracy_label = ttk.Label(self, text="Accuracy Score:")
        self.accuracy_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        
        self.accuracy_var = tk.StringVar(self)
        self.accuracy_output = ttk.Label(self, textvariable=self.accuracy_var)
        self.accuracy_output.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)
        
        self.data_label = ttk.Label(self, text="Enter data values (comma-separated):")
        self.data_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        
        self.data_entry = ttk.Entry(self)
        self.data_entry.grid(row=2, column=1, padx=10, pady=10, sticky=tk.W)
        
        self.predict_button = ttk.Button(self, text="Predict", command=self.predict)
        self.predict_button.grid(row=3, column=1, padx=10, pady=10, sticky=tk.E)


    def predict(self):
        """Trains and tests the selected ML model and makes and displays prediction based on user input data"""
        # Get the selected model
        model_name = self.model_var.get()
        
        # Load the dataset
        dataset = pd.read_csv('data.csv')
        X = dataset.iloc[:, 2:32].values
        Y = dataset.iloc[:, 1].values
        
        # Encode the labels
        labelencoder_Y = LabelEncoder()
        Y = labelencoder_Y.fit_transform(Y)
        
        # Split the dataset into training and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
        
        # Scale the data
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        # Fit the selected model
        if model_name == 'Logistic Regression':
            classifier = LogisticRegression(random_state = None)
        elif model_name == 'K-NN':
            classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        elif model_name == 'SVM':
            classifier = SVC(kernel = 'linear', random_state = None)
        elif model_name == 'K-SVM':
            classifier = SVC(kernel = 'rbf', random_state = None)
        elif model_name == 'Naive Bayes':
            classifier = GaussianNB()
        elif model_name == 'Decision Tree':
            classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = None)
        elif model_name == 'Random Forest':
            classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = None)
        
        # Fit the model on training data
        classifier.fit(X_train, Y_train)
        
        # Make predictions on test data
        Y_pred = classifier.predict(X_test)
        
        # Calculate the accuracy of the model
        accuracy = accuracy_score(Y_test, Y_pred)
        self.accuracy_var.set(round(accuracy, 2))  # Display accuracy score in GUI
        
        # Get user input
        input_data = self.data_entry.get()
        input_data = input_data.split(',')
        input_data = np.array(input_data).reshape(1, -1)
        input_data = input_data.astype(float)
        
        # Scale input data
        input_data = sc.transform(input_data)
        
        # Make prediction on user input data
        prediction = classifier.predict(input_data)
        if prediction == 0:
            prediction = "Benign"
        else:
            prediction = "Malignant"
        
        # Display prediction in GUI
        prediction_label = ttk.Label(self, text="Prediction:")
        prediction_label.grid(row=4, column=0, padx=10, pady=10, sticky=tk.W)
        prediction_output = ttk.Label(self, text=prediction)
        prediction_output.grid(row=4, column=1, padx=10, pady=10, sticky=tk.W)

if __name__ == "__main__":
    root = tk.Tk()
    splash = Splash(root)  # create the splash screen
    root.withdraw()  
    app = App()
    app.mainloop()