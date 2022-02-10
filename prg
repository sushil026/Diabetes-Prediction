import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
diabetes_dSet = pd.read_csv("/content/sample_data/diabetes.csv")
diabetes_dSet.head()
diabetes_dSet['Outcome'].value_counts()
x = diabetes_dSet.drop(columns = 'Outcome', axis= 1)
print(x)
y = diabetes_dSet['Outcome']
print(y)
sc = StandardScaler()
sc.fit(x)
stan_data = sc.transform(x)
print(stan_data)
x = stan_data
print(x)
print(y)
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.3, stratify=y, random_state=2)
print(X_test.shape, X_train.shape)
classifier = svm.SVC(kernel = 'linear')
classifier.fit(X_train,Y_train)
ip = (10,168,74,0,0,38,0.537,34)
npy_arr = np.asarray(ip)
reshaped = npy_arr.reshape(1,-1)
std_array = sc.transform(reshaped)
pred = classifier.predict(std_array)
if(pred[0] == 0):
  print("The person is not diabetic.")
else:
  print("The person is diabetic.")
  X_pred = classifier.predict(X_train)
accur_train = accuracy_score(X_pred, Y_train)
print("Accuracy for training data of our model is : ",accur_train)
X_pred = classifier.predict(X_test)
accur_test = accuracy_score(X_pred, Y_test)
print("Accuracy for testing data of our model is : ",accur_test)
// finish
