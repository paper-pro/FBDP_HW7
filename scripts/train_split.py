from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('iris.csv')
Y = df.iloc[:, -1]
X = df.iloc[:, 1:-1]
train_X, test_X, train_Y, test_Y = train_test_split(
    X, Y, test_size=0.3, random_state=0)
train_neo = train_X
train_neo['Species'] = train_Y
test_neo = test_X
train_neo.to_csv('train.csv', index=0, header=0)
test_neo.to_csv('test.csv', index=0, header=0)