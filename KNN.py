import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
class KNN:

    def __init__(self):
        self.neighbors_to_test = np.arange(1,21)

    def printReport(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y)

        best_k = None
        best_score = None
        best_model = None

        accuracies = []
        for k in self.neighbors_to_test:
            model = KNeighborsClassifier(n_neighbors=k)
            score = self.cross_validate_score(model, X_train, y_train)
            if(score > best_score or best_score == None):
                best_k = k
                best_score = score
                
                
        best_model = KNeighborsClassifier(n_neighbors=best_k)
        best_model.fit(X_train, y_train)
        print("Accuracy: {best_score}".format(best_score = best_model.score(X_test,y_test)))

        
    def cross_validate_score(self,model,X,y):
        kfold = KFold(n_splits=10)

        scores = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
            model.fit(X_train,y_train)
            score = model.score(X_test, y_test)
            scores.append(score)

        return np.mean(scores)