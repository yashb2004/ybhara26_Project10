"""
compare.py

Yash Bharadwaj
Spring 2023
CS152 

This program compares 7 ML models on their performance on cancer prognosis

To run this program type the following on the commandline:
python3 compare.py

"""
import tkinter as tk
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class AlgorithmComparisonGUI:
    def __init__(self, master):
        """Initializes the AlgorithmComparisonGUI object and assigns numerous attributes"""
        self.master = master
        master.title("Algorithm Comparison")

        # Create the widgets
        self.dataset_label = tk.Label(master, text="Compare ML models!")
        self.run_button = tk.Button(master, text="Run", command=self.run_algorithm_comparison)
        self.figure = Figure(figsize=(5, 4), dpi=200)
        self.canvas = FigureCanvasTkAgg(self.figure, master=master)

        # Add the widgets to the grid
        self.dataset_label.grid(row=0, column=0)
        self.run_button.grid(row=0, column=2)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=3)

    def run_algorithm_comparison(self):
        """plots a boxplot to compare the performance of the different ML algorithms"""
        
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

        # Define the models to compare
        models = []
        models.append(('LR', LogisticRegression(random_state = None)))
        models.append(('KNN', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)))
        models.append(('SVM', SVC(kernel = 'linear', random_state = None)))
        models.append(('K-SVM', SVC(kernel = 'rbf', random_state = None)))
        models.append(('NB', GaussianNB()))
        models.append(('DT', DecisionTreeClassifier(criterion = 'entropy', random_state = None)))
        models.append(('RFC', RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = None)))
        
        # Evaluate each model using cross-validation
        results = []
        names = []
        for name, model in models:
            kfold = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
            cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
            results.append(cv_results)
            names.append(name)

        # Create a boxplot of the results
        self.figure.clear()
        self.figure.suptitle('Algorithm Comparison')
        self.figure.subplots_adjust(bottom=0.2)
        ax = self.figure.add_subplot(111)
        ax.boxplot(results, labels=names)
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = AlgorithmComparisonGUI(root)
    root.mainloop()