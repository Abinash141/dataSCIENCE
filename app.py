import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle as pkl

df = pd.read_csv(r"C:\Users\abina\Documents\Data__\samples\heart_failure_clinical_records_dataset.csv")
df.head()
X = df["age"]
Y = df["platelets"]

train_x = X[:80]
train_y = Y[:80]

test_x = X[80:]
test_y = Y[80:] 


plt.scatter(X, Y)
plt.show() 


X_train, X_test, Y_train, Y_test= train_test_split(X,Y,random_state=42,test_size=0.2)   
df.score(X_train,Y_train)
