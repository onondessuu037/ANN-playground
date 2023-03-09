# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import seaborn as sns
import numpy as np



inputnumber = 8
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_Ans_df = pd.read_csv('sample_submission.csv')
print(type(train_df))
print(train_df)
combine = [train_df,test_df]
print(train_df.columns.values)
print(train_df.head())

print("Check isnull:")
print(train_df.isnull().sum())
print("Check isnull:")
print(test_df.isnull().sum())

# sns.pairplot(train_df,hue="Strength")
### plot the loss function
import matplotlib.pyplot as plt
# plt.show()


X_train = train_df.drop(["id","Strength"], axis=1).values### independent features
y_train = train_df["Strength"].values
X_test  = test_df.drop(["id"], axis=1).copy().values
y_test  = test_Ans_df.drop(["id"], axis=1).copy().values

print(type(X_train))
print(type(y_train))

#### Libraries From Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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







from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
print()
ilist = test_df.columns.tolist()
ilist.remove('id')
print(ilist)

#X, y = train_df[ilist].values, train_df['Strength'].values
X, y = X_train , y_train

sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
train_X, val_X, train_y, val_y = train_test_split(X, y, shuffle=True, test_size=0.2)
print(type(train_X))
print(val_X)
epochs=256
EPOCHS = 100
LEARNING_RATE = 0.0001
INPUT_SHAPE = train_X.shape[1]

print(INPUT_SHAPE)
#### Creating Modelwith Pytorch

class ANN_Model(nn.Module):
    def __init__(self,input_features=inputnumber,hidden1=64,hidden2=32,hidden3=20,out_features=1):
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

# class ANN_Model(nn.Module):
#     def __init__(self,INPUT_SHAPE=inputnumber,hidden1=64,hidden2=32,out_features=1):
#         super().__init__()
#         self.linear_layers = torch.nn.Sequential(
#             torch.nn.Linear(INPUT_SHAPE, hidden1),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden1, hidden2),
#             torch.nn.ReLU(),
#         )
#         self.output_layer = torch.nn.Linear(hidden2, out_features)
#     def forward(self, x):
#         x = self.linear_layers(x)
#         y = self.output_layer(x)
#         return y

    ####instantiate my ANN_model


torch.manual_seed(20)
torch_model = ANN_Model()
print("model.parameters")
print(torch_model.parameters)

###Backward Propogation-- Define the loss_function,define the optimizer
criterion=torch.nn.MSELoss()
optimizer = torch.optim.SGD(torch_model.parameters(), lr=LEARNING_RATE)
#optimizer=torch.optim.Adam(model.parameters(),lr=0.005)

train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_X, dtype=torch.float32),
    torch.tensor(train_y.reshape(-1, 1), dtype=torch.float32)
)
"""
<class 'torch.utils.data.dataset.TensorDataset'>
(tensor([-0.7426,  1.0621, -0.5838,  0.0337, -0.7218,  0.8167, -0.3420, -0.6393]), tensor([14.]))
(tensor([ 2.1400, -0.7027, -0.5838,  0.2119, -0.7218,  1.7241, -2.0100,  1.8321]), tensor([77.]))
(tensor([-0.7028, -0.7027, -0.5838, -0.2201, -0.7218,  1.5685,  0.7848, -0.3393]), tensor([39.]))
(tensor([-0.4745, -0.7027,  1.2258, -1.4569,  1.5270, -0.0545,  1.4962, -0.6964]), tensor([12.]))
...
"""
# print("********************")
# print(type(train_dataset))
# for i in train_dataset:
#     print(i)

#epochs = 256, train_dataset total number 4325, there are 17 sets in total
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=epochs,
    shuffle=True
)
"""
batch
<class 'torch.utils.data.dataloader.DataLoader'>
[tensor([[ 0.6011,  1.5632, -0.5838,  ..., -0.6132, -0.1959, -0.6393],
        [-1.2610, -0.7027,  2.4071,  ...,  0.8633,  0.1128, -0.3393],
        [-1.0279, -0.7027,  1.7075,  ...,  0.1815,  0.4164,  0.0607],
        ...,
        [-0.1058,  1.5992, -0.5838,  ..., -0.7778, -0.6786,  0.5464],
        [ 0.7186,  0.4219, -0.5838,  ..., -1.8136,  2.8123,  0.0607],
        [ 0.0268,  0.6641, -0.5838,  ..., -0.7778, -1.2859,  0.5464]]), tensor([[43.],
        [15.],
        [42.],
        [31.],
        [44.],
        [35.],
        ...
"""
# print("********************")
# print(type(train_loader))
# for i in train_loader:
#     print(len(i[0]),i)

train_loss_list = []
val_loss_list = []
train_metric_list = []
val_metric_list = []



for epoch in range(1, EPOCHS + 1):
    # training mode
    torch_model.train()
    train_loss = 0
    train_metric = 0
    for i, (batch_x, batch_y) in enumerate(train_loader, 1):
        # reset optimizer grad
        optimizer.zero_grad()
        # feed forward
        outputs = torch_model(batch_x)
        # get loss
        loss = criterion(outputs, batch_y)
        # get loss values from tensor
        train_loss += loss.item()
        # backward calculation
        loss.backward()
        # update weight parameters
        optimizer.step()
    # get average loss and metric
    train_loss = train_loss / i
    train_metric = np.sqrt(train_loss)  # root mean squared error

    # validation mode
    torch_model.eval()
    with torch.no_grad():
        outputs = torch_model(torch.tensor(val_X, dtype=torch.float32, device=device))
        loss = criterion(outputs, torch.tensor(val_y, dtype=torch.float32, device=device))
        val_loss = loss.item()  # / val_X.shape[0]
        val_metric = np.sqrt(val_loss)

    print(
        f'Epoch [{epoch}/{EPOCHS}], loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, root_mean_squared_error: {train_metric:.4f}, val_root_mean_squared_error: {val_metric:.4f}')
    train_loss_list.append(train_loss)
    train_metric_list.append(train_metric)
    val_loss_list.append(val_loss)
    val_metric_list.append(val_metric)

# plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='train')
# plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, label='val')
# plt.title('Loss History', fontsize=20)
# plt.xlabel('Epoch', fontsize=15)
# plt.ylabel('Loss', fontsize=15)
# plt.legend()
# plt.show
#
# plt.plot(range(1, len(train_metric_list) + 1), train_metric_list, label='train')
# plt.plot(range(1, len(val_metric_list) + 1), val_metric_list, label='val')
# plt.title('Validation Function History', fontsize=20)
# plt.xlabel('Epoch', fontsize=15)
# plt.ylabel('Root Mean Squared Error', fontsize=15)
# plt.legend()
# plt.show
#
# y_pred = torch_model(torch.tensor(val_X, dtype=torch.float32, device=device)).cpu().detach().numpy().reshape(-1)
# y_true = val_y
# print(y_pred)
# print(y_true)
# residual = y_pred - y_true
# plt.scatter(y_true, residual, s=3, alpha=0.5)
# plt.title('Plot Residual', fontsize=20)
# plt.xlabel('true price', fontsize=15)
# plt.ylabel('residual', fontsize=15)
# plt.show()



test_X = sc.transform(X_test)
test_X = torch.FloatTensor(test_X)
with torch.no_grad():
    test_result = torch_model(test_X)

print(test_result)

print(torch_model.eval())
print("model.parameters")
print(torch_model.parameters)
print("model.parameters")

for param in torch_model.parameters():
    print("****")
    print(param)


###*********************************************
# Strength = test_result[:,0].numpy()
# print(Strength)
# submission = pd.DataFrame({'id':  test_df.id, 'Strength': Strength})
# submission.to_csv('submission.csv', index=False)

###*********************************************




# final_losses=[]
# for i in range(epochs):
#     i=i+1
#     optimizer.zero_grad()
#
#     y_pred=model.forward(X_train)
#     # print("Epoch number: {} /// {} {}".format(i, y_pred, y_train))
#     loss=loss_function(y_pred,y_train)
#     final_losses.append(loss.item())
#     if i%10==1:
#         #print("Epoch number: {} and the loss : {} /// {} {}".format(i,loss.item(),y_pred,y_train))
#         print("Epoch number: {} and the loss : {} ".format(i,loss.item()))
#
#     loss.backward()
#     optimizer.step()
###--------------------------------------------




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
