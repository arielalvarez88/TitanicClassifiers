import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from AbstractAlgorithm import AbstractAlgorithm

class NeuralNetworks (AbstractAlgorithm):

    def printReport(self, X, y):
        model = self.get_new_model(X.shape[1])
        learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        results = {}

        y_binary = to_categorical(y, num_classes=2)
        for lr in learning_rates:
            model = self.get_new_model(X.shape[1])
            optimizer = SGD(lr=lr)
            model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
            
            model.fit(X,y_binary)
            score = model.evaluate(X, y_binary)        
            results['' + str(lr)] = score


        for lr, loss_acc in results.items():
            print('For Neural Networks With learning rate {lr} accuracy is {acc}'.format(lr=lr, acc=loss_acc[1]))

    def get_new_model(self, input_shape):
        model = Sequential()
        model.add(Dense(100, input_shape=(input_shape,), activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(25, input_shape=(input_shape,), activation='relu'))
        model.add(Dense(10, input_shape=(input_shape,), activation='relu'))
        model.add(Dense(2, activation='softmax'))
        return model
