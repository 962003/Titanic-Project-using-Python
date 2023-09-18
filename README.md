import seaborn as sns
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
Titanic=pd.read_csv("tested.csv")
Titanic.columns
Titanic.shape
Titanic.info()
Titanic['Survived'].value_counts()
Titanic.head()
Titanic.describe()
plt.figure(figsize=(6,6))
plt.title("number of survival")
plt.bar(list(Titanic['Survived'].value_counts().keys()),list(Titanic['Survived'].value_counts()),color=["r","y"])
plt.show()
Titanic.isnull().sum()
plt.figure(figsize=(6,6))
plt.title("Ticket Types")
plt.bar(list(Titanic['Pclass'].value_counts().keys()),list(Titanic['Pclass'].value_counts()),color=["g","r","b"])
plt.show()
Titanic['Pclass'].value_counts()
Titanic['Sex'].value_counts()
plt.figure(figsize=(6,6))
plt.title("Number of Males and Females Survived")
plt.bar(list(Titanic['Sex'].value_counts().keys()),list(Titanic['Sex'].value_counts()),color=["g","y"])
plt.show()
Titanic.head()
plt.figure(figsize=(5,6))
plt.hist(Titanic['Age'])
plt.title("Distribute Based on Age")
plt.xlabel("Age")
plt.show()
sum(Titanic['Age'].isnull())
sum(Titanic['Survived'].isnull())
Titanic=Titanic.dropna()
Titanic.isnull().sum()
a_train=Titanic[['Age']]
b_train=Titanic[['Survived']]
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(a_train,b_train)
a_tested=Titanic[['Age']]
b_pred=dtc.predict(a_tested)
b_pred
