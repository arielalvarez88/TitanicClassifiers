import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

df = pd.read_csv("train.csv")


features = ['Pclass','Sex','Age','SibSp','Parch','Fare']

Y = df['Survived']
x = df[features]

#Transform some columsn in categorical
x['Sex'] = x['Sex'].apply(lambda x: 0 if x.lower() == 'male' else 1 )
x['Sex'].astype('category')



def get_new_model():
    model = Sequential()
    model.add(Dense(100, input_shape=x.shape, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    return model

learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

results = {}
for lr in learning_rates:
    model = get_new_model()
    optimizer = SGD(lr=lr)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy")
    model.fit(x,Y)
    score = model.evaluate(x, Y)
    results['' + lr] = score 
