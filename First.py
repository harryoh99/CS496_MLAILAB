import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
os.chdir("/Users/jiooh/Documents/Jio/KAIST/2-2/개별연구/Data_Calibration")
x_data = torch.from_numpy(np.load("x.npy"))
y_data = torch.from_numpy(np.load("y.npy"))

#63560, 15, 667 for x
#63560, 15, 1 for y
#For year 1
year1_x = x_data[:,0,:]
year1_y = y_data[:,0,:]


#Training, test data split
year1_train_x = year1_x[:50000,:].float()
year1_train_y = year1_y[:50000,:].float()
year1_test_x = year1_x[50000:,:].float()
year1_test_y = year1_y[50000:, :].float()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(667,300)
        self.fc2 = nn.Linear(300,100)
        self.fc3 = nn.Linear(100,20)
        self.fc4 = nn.Linear(20,1)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

net = Net()
criterion = nn.MSELoss()
learning_rate = 0.01
iter_num=100
for idx in range(iter_num):
    for i in range(year1_train_x.shape[0]):
        input = year1_train_x[i]
        output = net(input)
        if(year1_train_y[i]==0):
            continue
        loss = criterion(output, year1_train_y[i])
        net.zero_grad()
        loss.backward()
        for f in net.parameters():
            f.data.sub_(f.grad.data * learning_rate)

#Testing period
loss_sum = 0
loss_sum2 = 0
for i in range(year1_test_x.shape[0]):
    input = year1_test_x[i]
    output = net(input)
    if(year1_test_y[i]==0):
        continue
    loss_sum += torch.abs(output-year1_test_y[i])/year1_test_y[i]
    loss_sum2 +=  criterion(output,year1_test_y[i])
print(loss_sum, loss_sum2)
