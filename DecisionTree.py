import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import KFold
from AbstractAlgorithm import AbstractAlgorithm

class DecisionTree(AbstractAlgorithm):
    """
    Creates a classifier using DecisionTree.
    """
    def __init__(self):
        self.neighbors_to_test = np.arange(1,21)

    def printReport(self,X,y):      
        print("Training DecisionTree...")
        model = tree.DecisionTreeClassifier()
        meanScore = self.cossValidiateScore(model,X,y)

        print("The mean accuracy for decision tree: " + str(meanScore))    
