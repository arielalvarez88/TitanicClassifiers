import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10)
df = pd.read_csv("train.csv")


features = ['Pclass', 'Sex','SibSp','Parch','Fare']

y = df['Survived']
X = df[features]

#Transform some columsn in categorical
X['Sex'] = X['Sex'].apply(lambda X: 0 if X.lower() == 'male' else 1 )
X['Sex'].astype('category')



def get_new_model():
    model = tree.DecisionTreeClassifier()    
    return model


#X['Age'] = X['Age'].fillna(X['Age'].mean())

kfold.get_n_splits(X)

scores = []
for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = get_new_model()    
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    scores.append(score)

print("The mean for decision tree: " + str(np.mean(scores)))    
