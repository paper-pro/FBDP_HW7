import pandas as pd
from sklearn.model_selection import train_test_split

res = pd.read_csv('part-r-00000.csv')#此处已把间隔符改为逗号并增加索引no,predict
res.sort_values("no", inplace=True)
df = pd.read_csv('iris.csv')
Y = df.iloc[:, -1]
X = df.iloc[:, 1:-1]
train_X, test_X, train_Y, test_Y = train_test_split(
         X, Y, test_size=0.3, random_state=0)
pred = list(res["predict"])
true = list(test_Y)
t = 0
p = 0
for i in range(len(pred)):
    if pred[i] == true[i]:
        t += 1
    else:
        p += 1
print("accuracy:", t/(t+p))
