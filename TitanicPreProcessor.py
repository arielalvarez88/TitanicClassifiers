import matplotlib.pyplot as plt
import pandas as pd

class TitanicPreProcessor:
    """
    Selects features, and preprocess the titanic dataset.
    """
    def __init__(self):
        self.features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

    def preprocess(self, df):
        """
        Preprocess the data and returns X and y.
        
        Extended description of function.

        Parameters
        ----------
        df : DataFrame
                
        Returns
        -------
        tuple
            X, y
        """
        y = df['Survived']
        X = df[self.features]
        # Transform some columsn in categorical
        X['Sex'] = X['Sex'].apply(lambda X: 0 if X.lower() == 'male' else 1)    
        X['Age'] = X['Age'].fillna(X['Age'].mean())

        featuresCount = len(self.features)
        pd.scatter_matrix(X, c=y, figsize = [featuresCount, featuresCount] , marker='.', s=150)
        plt.show()
        return X, y
