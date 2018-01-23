import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

class TitanicPreProcessor:
    """
    Selects features, and preprocess the titanic dataset.
    """
    def __init__(self):
        self.features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']

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
        embarkedMode = df['Embarked'].mode()[0]
        X['Embarked'] = X['Embarked'].fillna(embarkedMode)
        X['Embarked'] = X['Embarked'].astype('category').cat.codes.astype('category')

        featuresCount = len(self.features)
        #pd.scatter_matrix(X, c=y, figsize = [featuresCount,featuresCount] , marker='.', s=150)
        #plt.show()

        #self.visualizeWithTSNE(X,y)
        #self.visualizeWithPCA(X,y)
        return X, y

    def visualizeWithTSNE(self,X,y):
        plt.figure()
        plt.title("TSNE")
        tsne = TSNE().fit_transform(X)
        plt.scatter(tsne[:,0], tsne[:,1],c=y)
        plt.show()

    def visualizeWithPCA(self,X,y):
        plt.figure()
        pca = PCA(2)
        plt.title("PCA")
        transformedVals = pca.fit_transform(X,y)
        plt.scatter(transformedVals[:,0], transformedVals[:,1],c=y)
        plt.show()
