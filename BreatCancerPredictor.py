import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
df = pd.read_csv('/Users/a./Downloads/datasets_180_408_data.csv')
df = df.drop(['Unnamed: 32'],axis=1)
df['diagnosis'].replace('M','0',inplace=True)
df['diagnosis'].replace('B','1',inplace=True)
x = df.drop(['diagnosis'],axis=1)
y = df[['diagnosis']]
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.5)
# sns.pairplot(df, hue='diagnosis', vars = ['radius_mean','texture_mean','perimeter_mean',
#        'area_mean', 'smoothness_mean'])
# plt.figure(figsize=(20,12))
# sns.heatmap(df.corr(),annot=True)
# plt.show()
svm = SVC()
svm.fit(X_train,y_train)
print(svm.score(X_train,y_train))
print(svm.score(X_test,y_test))
y_pre= svm.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pre))
# Achieving only 30% accuracy
# now will try to scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.fit_transform(X_test)
svm.fit(x_train_scaled,y_train)
ypred = svm.predict(x_test_scaled)
print(classification_report(y_test,ypred))
# Now achieved 95 % accuracy in predicting breast Cancer by scaling the Input Values
