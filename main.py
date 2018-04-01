import pandas as pd
from TitanicPreProcessor import TitanicPreProcessor
from DecisionTree import DecisionTree
from SVM import SVM
from KNN import KNN

df = pd.read_csv("train.csv")
preProc = TitanicPreProcessor()

inputIsValid = True

while True:
    wantsEDA = input("What to see EDA ? (Y/N)")
    wantsEDA = wantsEDA.lower()
    inputIsValid = wantsEDA == "y" or wantsEDA == "n"
    if inputIsValid:
        break
    else:
        "Invalid input."

if wantsEDA == "y":
    preProc.showEDA()


X,y = preProc.preprocess(df)


classifiers = [    
    "KNN",
    "DecisionTree",
    "SVM",
    "NeuralNetworks"
]

while(True):
    print("Select a list classifiers by entering a comma separated list of numbers represeting these algorithms: ")

    for number, algorithm in enumerate(classifiers):
        print("For {algorithm} type {number}.".format(algorithm=algorithm, number=number))

    userChoice = input("")
    userChoice = userChoice.split(",")
    algorithmsChosen = []
    inputIsValid = True
    for idx in userChoice:
        try:
            algorithmName = classifiers[int(idx)]
            algorithmsChosen.append(algorithmName)
        except:
            inputIsValid = False
            print("Invalid input.")

    if not inputIsValid:
        continue
    for className in algorithmsChosen:

        module = __import__(className)
        class_ = getattr(module, className)
        algorithm = class_()
        algorithm.printReport(X,y)


