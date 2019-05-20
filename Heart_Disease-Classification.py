import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('heart.csv')

features=dataset.iloc[:,:-1].values

labels=dataset.iloc[:,-1].values


#if some data is missing from the dataset then

from sklearn.impute import SimpleImputer

#make an imputer object
#we can replace missing values by 'NaN'
#statergy for missing feature having integer values is take the man and insert
#axis=0-column(mean of values in column),   1-row

imputer=SimpleImputer(missing_values=np.nan,strategy='mean')

#fit your x matrix in imputer
imputer.fit(features)

features=imputer.transform(features)

#In case you have string as one or more of your features
#encode them using LabelEncoder

#from sklearn.preprocessing import LabelEncoder
#label_encoder_X=LabelEncoder()
#X[:,column_index]=label_encoder_X.fit_transform(X[:,column_index])

#As models use mathematical computations on data ,after LabelEncoding our data will be converted to integers
#and integers may be compared by the model,to solve this we make seperate column for each integer value
#and place "1" where the integer value matches ,this is called one hot-encoding
#suppose one of your feature is country in which you have three countries France,Germany,Ireland
#after label-encoding you'll have 0-France,1-Germany,2-Ireland
#after one hot-encoding you'll have three features instead of a country feature
#namely, France Germany Ireland
#           1       0       0
#           0       1       0
#           0       0       1

#from sklearn.preprocessing import OneHotEncoder
#onehotencoder=OneHotEncoder(categorcial_features=[column_index])
#X=onehotencoder.fit_transform(x).toarray()

#The same can be applied to the target/labels
#for only 2 different value of labels we can use labelsencoder only
#for more than two use onehotencoder

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(features,labels,test_size=0.25,random_state=42)

#feature scaling
#we do feature scaling because on computing euclidian distance or any error the bigger feature values
#i.e the features having large values like salary feature,
#dominate the error ,that is because the (y2-y1)^2 term dominates (x2-2x1)^2 term so,
#the answer is approximately equal to (y2-y1) i.e the difference in bigger value feature
#so ,we scale all our feature so that all feature will have same weightage on the decision and error

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)


# from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
# from sklearn.model_selection import GridSearchCV

# param_grid={
#     # 'n_estimators':[40],
#     # 'max_depth':[10]
#     # 'n_neighbors':[5,10,15]
#     # 'min_samples_split':[20,40]
#     # 'C':[50,100,150,200,250,300,500,1000]
# }

# clf=GridSearchCV(SVC(kernel='rbf',gamma='scale'),param_grid)

# clf=GridSearchCV(DecisionTreeClassifier(),param_grid)
clf=DecisionTreeClassifier(min_samples_split=20)
clf.fit(X_train_scaled,Y_train)
pred=clf.predict(X_test)

# print(pred)
# print(clf.best_params_)
print(accuracy_score(pred,Y_test),f1_score(Y_test,pred))

# print(X_train_scaled[0])
# plt.scatter(X_train[:,0],X_train[:,[4]])
# # plt.scatter(X_test[:,0],X_test[:,[4]],color='r')
# plt.xlabel("Age")
# plt.ylabel("cholestrol")
# plt.show()

