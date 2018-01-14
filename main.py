import pandas as pd
from TitanicPreProcessor import TitanicPreProcessor
from KNN import KNN



df = pd.read_csv("train.csv")
preProc = TitanicPreProcessor()
X,y = preProc.preprocess(df)

kNN = KNN()
print(kNN.printReport(X,y))

#classifiers = {
#    "Decision Tree": 1,
#    "KNN": 2
#}



#print("Select a classifier: ")

#for algorithm, number in classifiers.iteritems():
#    print("For {algorithm} type {number}")

#option = 



