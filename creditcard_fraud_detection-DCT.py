#Credit Card Fraud Detection
#Decission Tree Classifier method

import pandas as pd
import numpy as np

#created dataframe
df=pd.read_csv('creditcard.csv')

#dropped columns that are not required
df.drop(['Time','Amount'],axis=1,inplace=True)

#splitting data for training
x=df.drop('Class',axis=1)
y=df['Class']

#checking dataframe for null values
print('\nChecking null values in data set...\n')
df.info() #{or alternate code 'df.isnull().sum()'}

#dropping duplicate values
df.drop_duplicates(inplace=True)

#generating train/test data individually for both splitted data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

#create model
from sklearn.tree import DecisionTreeClassifier
cf=DecisionTreeClassifier()
cf.fit(x_train,y_train)
y_pred=cf.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print('\n*****Classification Report*****')
print(classification_report(y_test, y_pred))
print('*****Confusion Matrix*****')
print(confusion_matrix(y_test, y_pred))

from sklearn import metrics
m=metrics.mean_absolute_error(y_test, y_pred)
print('\nMean absolute error check:',np.sqrt(m))

#working of model
print('\nTransaction codes: 0.503302,0.930065139646533,-0.857525171542855,2.04294011557666,-1.50594639143412,\n-1.00018456183002,-1.99136333313202,0.460576746896755,-1.12410116690148,-1.97461696232863,\n2.94606281176549,-2.1498630541179,-0.646459994434482,-2.23883094451216,0.291703946709641,\n-2.21330281047888,-5.40001437327305,-1.44538878051743,0.36806011087656,0.166740829023922,\n0.379924772081051,-0.0621911206869668,-0.0121869727427656,0.479787698400125,0.531946545714183,\n-0.441322928572273,0.460792336235155,0.219985374658915')
output=cf.predict([[0.503302,0.930065139646533,-0.857525171542855,2.04294011557666,-1.50594639143412,-1.00018456183002,-1.99136333313202,0.460576746896755,-1.12410116690148,-1.97461696232863,2.94606281176549,-2.1498630541179,-0.646459994434482,-2.23883094451216,0.291703946709641,-2.21330281047888,-5.40001437327305,-1.44538878051743,0.36806011087656,0.166740829023922,0.379924772081051,-0.0621911206869668,-0.0121869727427656	,0.479787698400125,0.531946545714183,-0.441322928572273,0.460792336235155,0.219985374658915]])
print('\nPredicted code:',output)
if output==0:
    print('\n"Fraud Transaction"')
else:
    print('\n"Valid Transaction"')
    
print('\nAccuracy score:{}\n'.format(accuracy_score(y_test,y_pred)))