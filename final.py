import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
os.chdir("/Users/jiooh/Documents/Jio/KAIST/2-2/개별연구/Data_Calibration")

class re_data(Dataset):
    def __init__(self,x_data, y_data):
        indice = y_data.nonzero()[:,0]
        self.data = x_data[indice]
        self.labels = y_data[indice]
        self.len = self.data.shape[0]
    
    def __getitem__(self,index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return self.len

def save_data():
    torch.save(model, 'crt1_model.pt')
    torch.save(model2, 'crt2_model.pt')

def load_data():
    model = torch.load('crt1_model.pt')
    model2 = torch.load('crt2_model.pt')
    return model, model2


#63560, 15, 667 for x
#63560, 15, 1 for y
#For year 1

#Model
model = nn.Sequential(
    nn.Linear(667,300),
    nn.ReLU(),
    nn.Linear(300,100),
    nn.ReLU(),
    nn.Linear(100,20),
    nn.ReLU(),
    nn.Linear(20,1)
)
model2 = nn.Sequential(
    nn.Linear(667,300),
    nn.ReLU(),
    nn.Linear(300,100),
    nn.ReLU(),
    nn.Linear(100,20),
    nn.ReLU(),
    nn.Linear(20,1)
)

#Criterion
criterion = nn.MSELoss()

def criterion2(output, target):
    res = torch.abs(output-target)/target
    return res.mean()

#DATA LOAD/SPLIT
x = torch.from_numpy(np.load("x.npy"))
y = torch.from_numpy(np.load("y.npy"))

x_data = x[:,0,:].float()
y_data = y[:,0,:].float()
for i in range(1,15):
    temp_x = x[:,i,:].float()
    temp_y = y[:,i,:].float()
    x_data = torch.cat((x_data, temp_x), dim=0)
    y_data = torch.cat((y_data, temp_y), dim=0)

dataset_size = x_data.shape[0] #(63560*15,667)
validation_split = 0.1

indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_x = x_data[train_indices]
train_y = y_data[train_indices]

test_x = x_data[val_indices]
test_y = y_data[val_indices]

train_dataset = re_data(train_x,train_y)
test_dataset = re_data(test_x,test_y)
batch_size = 50

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle = True)

learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
optimizer2 = optim.Adam(model2.parameters(), lr = learning_rate)
num_epoch = 10

#Training loss saved here
training_loss =[]
training2_loss =[]

for epoch in range(num_epoch):
    #Training
    for batch_index, (data,target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        training_loss.append([epoch,round(loss.item(),3)])

for epoch in range(num_epoch):
    #Training
    for batch_index, (data,target) in enumerate(train_loader):
        optimizer2.zero_grad()
        output = model2(data)
        loss = criterion2(output, target)
        loss.backward()
        optimizer2.step()
        training2_loss.append([epoch,round(loss.item(),3)])


save_data()
model,model2 = load_data()
#Tested with criterion 2
loss21_list = []
loss22_list = []

with torch.no_grad():
    for batch_index, (data,target) in enumerate(validation_loader):
        output1 = model(data) #Trained with criterion1
        output2 = model2(data) #Trained with criterion 2

        loss21 = criterion2(output1,target) #TRained with criterion 1, tested with 2
        loss22 = criterion2(output2,target) #Trainde with criterion 2, tested with 2
        loss21_list.append(loss21.item())
        loss22_list.append(loss22.item())

loss21_avg = np.average(loss21_list)
loss22_avg = np.average(loss22_list)

#Write the average
f = open("output.txt","w")
f.write('Training Loss with criterion 1: \n')
f.write(" ".join(training_loss.__str__()))

f.write("\nTraining Loss with criterion 2: \n")
f.write(" ".join(training2_loss.__str__()))

f.write("\nAverage loss of model trained with criterion 1\n")
f.write(str(loss21_avg))
f.write("\nAverage loss of model trained with criterion 2\n")
f.write(str(loss22_avg))

f.close()



plt.figure()
plt.title("Test Results on each training")
plt.xlabel("test count")
plt.ylabel("loss")
plt.plot(loss21_list, marker = 'o', label = 'Trained with crt 1, test with crt 2', color = 'blue')
plt.plot(loss22_list, marker = 'x', label = 'Trained with crt 2, test with crt 2', color = 'pink')
plt.legend() 
plt.savefig('test_result.png')
plt.close()



