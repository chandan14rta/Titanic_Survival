#import Libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

#ignore warnings
warnings.filterwarnings('ignore')

#Load train and test files
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#Fill nan values
train['Age'].fillna(train['Age'].median(),inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)

#Label encoding
le=LabelEncoder()
train['Sex']=le.fit_transform(train['Sex'])
train['Embarked']=le.fit_transform(train['Embarked'])

#create x and y variable for model
train_x=train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
train_y=train['Survived']

#create random forest model with 100 estimators and fit with x and y
model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_x,train_y)

#similar to train
test['Age'].fillna(test['Age'].median(),inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0],inplace=True)
test['Fare'].fillna(test['Fare'].median(),inplace=True)
test['Sex']=le.fit_transform(test['Sex'])
test['Embarked']=le.fit_transform(test['Embarked'])
test_x=test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

#predict for test
test_y=model.predict(test_x)

#print the prediction data
submission=pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':test_y
})
print(submission)
