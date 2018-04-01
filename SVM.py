import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold, train_test_split
from AbstractAlgorithm import AbstractAlgorithm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

class SVM(AbstractAlgorithm):
    """
    Creates a classifier using DecisionTree.
    """
    def __init__(self):
        self.neighbors_to_test = np.arange(1,21)
        
        self.gridSearchParams = [
            {'C': np.logspace(-3,3,7), 'kernel':['poly','linear']}
        ]
        self.scaler = StandardScaler()

    def printReport(self,X,y):      
        model = SVC()
        print("Training SVM...")
        X_train, X_test, y_train , y_test = train_test_split(X,y,test_size=0.3)
        X_train  = self.scale(X_train)
        X_test  = self.scale(X_test)        

        gridSearch = GridSearchCV(model, self.gridSearchParams,cv=10)
        gridSearch.fit(X_train,y_train)
        score = gridSearch.score(X_test,y_test)    
        print("The accuracy for SVM: " + str(score))    

    def scale(self,X):        
        return self.scaler.fit_transform(X)        