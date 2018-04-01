import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import seaborn as sns



class TitanicPreProcessor:
    """
    Selects features, and preprocess the titanic dataset.
    """

    def __init__(self):

        self.uncoverted_features = ['Sex', 'Pclass', 'Embarked', 'Title',
                                    'SibSp', 'Parch', 'Age', 'Fare', 'IsAlone']  # pretty name/values

        self.features = ['Pclass', 'Sex_Code', 'SibSp', 'Parch', 'Title_Code',
                         'Embarked_Code', 'Age', 'FamilySize', 'IsAlone', 'AgeBin_Code', 'FareBin_Code']

        self.target_feature = ['Survived']

        self.data1_x_bin = ['Sex_Code', 'Pclass', 'Embarked_Code',
                            'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']

        self.data1_x = ['Sex', 'Pclass', 'Embarked', 'Title', 'SibSp', 'Parch',
                        'Age', 'Fare', 'FamilySize', 'IsAlone']  # pretty name/values for charts
        self.data1_x_calc = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code',
                             'SibSp', 'Parch', 'Age', 'Fare']  # coded for algorithm calculation
        self.data1_xy = self.target_feature + self.data1_x
        print('Original X Y: ', self.data1_xy, '\n')

        self.data = None
        self.raw_data = None

        self.X = None
        self.y = None

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
        self.data = df
        self.raw_data = df.copy(deep=True)
        df_to_clean = [self.data, self.raw_data]

        # The 4 C's of Machine Learning/Data Science
        for df in df_to_clean:
            self.correct(df)
            self.completeData(df)
            self.createFeatures(df)
            self.convert(df)

        featuresCount = len(self.features)

        self.data_integrity_report(self.data)

        self.y = df['Survived']
        self.X = df[self.features]
        
        return self.X, self.y

    def showEDA(self):
        self.exploratory_data_analysis(self.data,self.X,self.y)


    def convert(self, X):
        """
            Format the data. Primarily it converts categorical columns in formats like int64, string, etc. to category type.
        """
        label = LabelEncoder()
        X['Sex_Code'] = label.fit_transform(X['Sex'])
        X['Embarked_Code'] = label.fit_transform(X['Embarked'])
        X['Title_Code'] = label.fit_transform(X['Title'])
        X['AgeBin_Code'] = label.fit_transform(X['AgeBin'])
        X['FareBin_Code'] = label.fit_transform(X['FareBin'])

    def createFeatures(self, X):
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = 1
        X['IsAlone'][X['FamilySize'] > 1] = 0
        X['Title'] = X['Name'].str.split(", ", expand=True)[
            1].str.split('.', expand=True)[0]
        notCommonTitles = X['Title'].value_counts() < 10
        X['Title'].apply(
            lambda x: 'Misc' if notCommonTitles.loc[x] == True else x)
        X['FareBin'] = pd.qcut(X['Fare'], 4)
        X['AgeBin'] = pd.cut(X['Age'].astype(int), 5)

        # self.raw_data['Title'].apply(lambda x : 'Misc' of len())

    def visualizeWithTSNE(self, X, y):
        plt.figure()
        plt.title("TSNE")
        tsne = TSNE().fit_transform(X)
        plt.scatter(tsne[:, 0], tsne[:, 1], c=y)
        plt.show()

    def visualizeWithPCA(self, X, y):
        plt.figure()
        pca = PCA(2)
        plt.title("PCA")
        transformedVals = pca.fit_transform(X, y)
        plt.scatter(transformedVals[:, 0], transformedVals[:, 1], c=y)
        plt.show()

    def completeData(self, X):
        # Transform some columsn in categorical
        X['Age'].fillna(X['Age'].mean(), inplace=True)
        embarkedMode = X['Embarked'].mode()[0]
        X['Embarked'] = X['Embarked'].fillna(embarkedMode)

    def correct(self, X):
        """
            Data is in pretty good shape, we can skip this step.
        """
        pass

    def data_integrity_report(self, data):
        print("Null counts: ")
        print(data.isnull().sum())
        print("-" * 10)

        print("Overview: ")
        print(data.info())
        print("-" * 10)

    def exploratory_data_analysis(self, data,X,y):
        # Discrete Variable Correlation by Survival using
        # group by aka pivot table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
        for x in self.data1_x:
            if data[x].dtype != 'float64':
                print('Survival Correlation by:', x)
                print(data[[x, self.target_feature[0]]].groupby(
                    x, as_index=False).mean())
                print("Ariel")
                print(data[[x, self.target_feature[0]]].groupby(
                    x, as_index=False).mean())
                print('-' * 10, '\n')

        # using crosstabs:                   https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html
        print(pd.crosstab(data['Title'], data[self.target_feature[0]]))

        self.show_graphs(data,X,y)

    def show_graphs(self, data, X, y):

        featuresCount =  len(self.features)

        pd.scatter_matrix(X, c=y, figsize = [featuresCount,featuresCount] , marker='.', s=150)
        plt.show()

        self.visualizeWithTSNE(X,y)
        self.visualizeWithPCA(X,y)
        
        self.show_graphs_from_kaggle_kernel(data)
        plt.show()


    def show_graphs_from_kaggle_kernel(self,data):
        plt.figure(figsize=[16, 12])

        plt.subplot(231)
        plt.boxplot(x=data  ['Fare'], showmeans=True, meanline=True)
        plt.title('Fare Boxplot')
        plt.ylabel('Fare ($)')

        
        plt.subplot(232)
        plt.boxplot(data['Age'], showmeans=True, meanline=True)
        plt.title('Age Boxplot')
        plt.ylabel('Age (years)')

        plt.subplot(233)
        plt.boxplot(data['FamilySize'], showmeans = True, meanline = True)
        plt.title('Family Size Boxplot')
        plt.ylabel('Family Size (#)')

        plt.subplot(234)
        plt.hist(x = [ data[ data['Survived'] == 1]['Fare'], data[ data['Survived'] == 0]['Fare']   ], stacked=True, color=['g','r'], label=['Survived','Dead'])
        plt.title("Fare hist by Survival")
        plt.xlabel("Fare ($)")
        plt.ylabel("# of passanger")
        plt.legend()

        plt.subplot(235)
        plt.hist(x = [data[data['Survived']==1]['Age'], data[data['Survived']==0]['Age']], 
                stacked=True, color = ['g','r'],label = ['Survived','Dead'])
        plt.title('Age Histogram by Survival')
        plt.xlabel('Age (Years)')
        plt.ylabel('# of Passengers')
        plt.legend()



        plt.subplot(236)
        plt.hist(x = [data[data['Survived']==1]['FamilySize'], data[data['Survived']==0]['FamilySize']], 
                stacked=True, color = ['g','r'],label = ['Survived','Dead'])
        plt.title('Family Size Histogram by Survival')
        plt.xlabel('Family Size (#)')
        plt.ylabel('# of Passengers')
        plt.legend()


        fig, saxis = plt.subplots(2,3,figsize=(16,12))
        sns.barplot(x='Embarked', y='Survived', data=data, ax=saxis[0,0])
        sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data, ax = saxis[0,1])
        sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data, ax = saxis[0,2])


        sns.pointplot(x = 'FareBin', y = 'Survived',  data=data, ax = saxis[1,0])
        sns.pointplot(x = 'AgeBin', y = 'Survived',  data=data, ax = saxis[1,1])
        sns.pointplot(x = 'FamilySize', y = 'Survived', data=data, ax = saxis[1,2])

        fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))

        sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = data, ax = axis1)
        axis1.set_title('Pclass vs Fare Survival Comparison')

        sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data, split = True, ax = axis2)
        axis2.set_title('Pclass vs Age Survival Comparison')

        sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = data, ax = axis3)
        axis3.set_title('Pclass vs Family Size Survival Comparison')

        sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = data, ax = axis1)
        axis1.set_title('Pclass vs Fare Survival Comparison')

        sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data, split = True, ax = axis2)
        axis2.set_title('Pclass vs Age Survival Comparison')

        sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = data, ax = axis3)
        axis3.set_title('Pclass vs Family Size Survival Comparison')
        plt.show()


        
