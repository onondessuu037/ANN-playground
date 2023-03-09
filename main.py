# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
inputnumber = 12
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_Ans_df = pd.read_csv('gender_submission.csv')
print(type(train_df))
print(train_df)
combine = [train_df,test_df]
print(train_df.columns.values)
print(train_df.head())

print("Check isnull:")
print(train_df.isnull().sum())
print("Check isnull:")
print(test_df.isnull().sum())

mean_age = train_df['Age'].mean()
train_df['Age'].fillna(value=mean_age, inplace=True)
mean_age = test_df['Age'].mean()
test_df['Age'].fillna(value=mean_age, inplace=True)

print("Check isnull:")
print(train_df.isnull().sum())
print("Check isnull:")
print(test_df.isnull().sum())

# for dataset in combine:
#     dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
#     dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
#     dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
#     dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
#     dataset.loc[ dataset['Age'] > 64, 'Age']
# print("Check isnull:")
# print(dataset.isnull().sum())
#print(combine)


modes = train_df.mode().iloc[0]            # Need to take the first elements because there can be multiple modes
train_df.fillna(modes, inplace=True)       # Replace NaNs
modes = test_df.mode().iloc[0]            # Need to take the first elements because there can be multiple modes
test_df.fillna(modes, inplace=True)       # Replace NaNs

print("Check isnull:")
print(train_df.isnull().sum())
print("Check isnull:")
print(test_df.isnull().sum())
import seaborn as sns
import numpy as np

# df['gender']=np.where(df['gender']=="Female",1,0)  #1:Female 0:Male
# df['country']=np.where(df['country']=="France",0,df['country'])  #['France', 'Spain', 'Germany']
# df['country']=np.where(df['country']=="Spain",1,df['country'])  #['France', 'Spain', 'Germany']
# df['country']=np.where(df['country']=="Germany",2,df['country'])  #['France', 'Spain', 'Germany']
# mylist = []
# for i in df['country']:
#     if i not in mylist:
#         mylist.append(i)
# print(mylist)

print(train_df['Ticket'])
print(train_df['Fare'])
print(train_df.info())

sns.pairplot(train_df,hue="Survived")
### plot the loss function
import matplotlib.pyplot as plt

plt.show()

# One-hot encoding data
def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=prefix)
    #dummies = pd.get_dummies(df,columns=[column])
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df

train_df = onehot_encode(train_df, 'Sex','Sex')
train_df = onehot_encode(train_df, 'Pclass','Pclass')
train_df = onehot_encode(train_df, 'Embarked','Embarked')
test_df = onehot_encode(test_df, 'Sex','Sex')
test_df = onehot_encode(test_df, 'Pclass','Pclass')
test_df = onehot_encode(test_df, 'Embarked','Embarked')

print(train_df.head())



X_train = train_df.drop(["Survived","PassengerId","Name","Ticket","Cabin"], axis=1).values### independent features
y_train = train_df["Survived"].values
X_test  = test_df.drop(["PassengerId","Name","Ticket","Cabin"], axis=1).copy().values
y_test  = test_Ans_df.drop(["PassengerId"], axis=1).copy().values
#### Libraries From Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

print("********************")
print(type(X_train))
print(type(y_train))
print(X_train)
print(y_train)
print(train_df.info())
##### Creating Tensors
X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)
# y_train=torch.tensor(y_train, requires_grad=True).type(torch.float32)
# y_test=torch.tensor(y_test, requires_grad=True).type(torch.float32)

print(X_train.shape)
print(type(X_train))
print(type(y_train))
"""
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
"""
print(f"X_train {X_train}")
"""
tensor([[7.0000e+00,...
"""
print(f"y_train {y_train}")


#### Creating Modelwith Pytorch

class ANN_Model(nn.Module):
    def __init__(self,input_features=inputnumber,hidden1=10,hidden2=10,hidden3=20,out_features=2):
        super().__init__()
        self.f_connected1=nn.Linear(input_features,hidden1)
        self.f_connected2=nn.Linear(hidden1,hidden2)
        #self.f_connected3=nn.Linear(hidden2,hidden3)
        self.out=nn.Linear(hidden2,out_features)
    def forward(self,x):
        x=F.relu(self.f_connected1(x))
        x=F.relu(self.f_connected2(x))
        #x=F.relu(self.f_connected3(x))
        x=self.out(x)
        #x = F.log_softmax(self.out(x), dim=1)
        return x
####instantiate my ANN_model
torch.manual_seed(20)
model=ANN_Model()
print("model.parameters")
print(model.parameters)

###Backward Propogation-- Define the loss_function,define the optimizer
loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.005)

epochs=5000
final_losses=[]
for i in range(epochs):
    i=i+1
    y_pred=model.forward(X_train)
    # print("Epoch number: {} /// {} {}".format(i, y_pred, y_train))
    loss=loss_function(y_pred,y_train)
    final_losses.append(loss.item())
    if i%10==1:
        #print("Epoch number: {} and the loss : {} /// {} {}".format(i,loss.item(),y_pred,y_train))
        print("Epoch number: {} and the loss : {} ".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
###--------------------------------------------



### plot the loss function
import matplotlib.pyplot as plt

plt.plot(range(epochs),final_losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

#### Prediction In X_test data
predictions=[]
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_pred=model(data)
        #print(y_pred.argmax()) #y_pred.argmax() find out the index of maximum value
        predictions.append(y_pred.argmax().item())
        #print(y_pred.argmax().item())

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predictions)
print("cm")
print(cm)
# print("Anser:")
# print(y_test)
# print(predictions)

plt.figure(figsize=(10,6))
sns.heatmap(cm,annot=True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,predictions)
print(score)

print(model.eval())
print("model.parameters")
print(model.parameters)
print("model.parameters")

for param in model.parameters():
    print("****")
    print(param)

###*********************************************
X_test_var = torch.FloatTensor(X_test)
with torch.no_grad():
    test_result = model(X_test_var)
values, labels = torch.max(test_result, 1)
survived = labels.data.numpy()
submission = pd.DataFrame({'PassengerId':  test_df.PassengerId, 'Survived': survived})
submission.to_csv('submission.csv', index=False)

###*********************************************

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
