from abc import ABCMeta, abstractmethod
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

class AbstractAlgorithm():
    __metaclass__ = ABCMeta

    @abstractmethod
    def printReport(self, X, y):
        pass

    def cossValidiateScore(self,model,X,y):
        kfold = KFold(n_splits=10)

        scores = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
            model.fit(X_train,y_train)
            score = model.score(X_test, y_test)
            scores.append(score)

        return np.mean(scores)    

