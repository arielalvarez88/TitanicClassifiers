import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from AbstractAlgorithm import AbstractAlgorithm

class KNN(AbstractAlgorithm):
    """
    Creates a classifier using K Nearest Neighbors.
    """
    def __init__(self):
        self.neighbors_to_test = np.arange(1,21)

    def printReport(self, X, y):
        """
        Prints Accuracy of the best model.

        Params
        -----------------
        X : pandas.Dataframe
        y : pandas.Dataframe

        Returns
        -----------------
        None

        """
        print("Training KNN...")
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y)

        best_k = None
        best_score = None
        best_model = None

        accuracies = []
        for k in self.neighbors_to_test:
            model = KNeighborsClassifier(n_neighbors=k)
            score = self.cossValidiateScore(model, X_train, y_train)
            if(best_score == None or score > best_score):
                best_k = k
                best_score = score
                
                
        best_model = KNeighborsClassifier(n_neighbors=best_k)
        best_model.fit(X_train, y_train)
        print("Accuracy for KNN: {best_score}".format(best_score = best_model.score(X_test,y_test)))

   