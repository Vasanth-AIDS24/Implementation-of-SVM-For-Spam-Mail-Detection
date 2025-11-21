# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Vasanth P
RegisterNumber:212224230295  
*/



import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data
data.info
data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
![SVM For Spam Mail Detection](sam.png)

```
Data:
```

<img width="917" height="502" alt="image" src="https://github.com/user-attachments/assets/59160345-2803-48cc-8f1b-4b0820e1a325" />

```
Confusion Matrix:
```


<img width="125" height="48" alt="image" src="https://github.com/user-attachments/assets/bd34e23c-8c45-4042-9f65-59dbdc1ccfde" />

```
Classification:
```


<img width="602" height="221" alt="image" src="https://github.com/user-attachments/assets/dd74d190-be8d-4906-bdcb-bdf5d8aef671" />

```
Accuracy:
```


<img width="237" height="38" alt="image" src="https://github.com/user-attachments/assets/93a3abe7-0d34-47fc-8557-7e02227fe0fc" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
