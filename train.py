import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = np.loadtxt('training_set_no.csv', delimiter=',')
test = np.loadtxt('testing_set_no.csv', delimiter = ',')
l =[]
columns = ['svm','logistic_regression','random_forrest']
#l.append(test[:,0])

X = data[:, 1:]
y = data[:, 0].astype(np.int)

##SVM

clf = svm.SVC()
clf.fit(X,y)

#print(clf.predict([[0,0,0,0,0.0001870907,0.0001920123,0.3333333333]]))
l.append(clf.predict(test[:,:]))


## LR
lr = LogisticRegression()
lr.fit(X,y)
l.append(lr.predict(test[:,:]))


## Random Forest
rfClf = RandomForestClassifier()
rfClf = rfClf.fit(X,y)
l.append(rfClf.predict(test[:,:]))


#df = pd.DataFrame(data=l, columns=columns)
df = pd.DataFrame.from_items(zip(columns, l))
#print (df)

df.to_csv('test_output.csv', sep=',', encoding='utf-8')







#df = pd.read_csv('testing_set.csv', delimiter = ',')
#df = df.assign(output=prediction)
#print(df)

#np.savetxt('test_output.csv',prediction,delimiter=',')

#print (test)
