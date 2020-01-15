import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

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


#63560, 15, 667 for x
#63560, 15, 1 for y
#For year 1

#Training, test data split
model = nn.Sequential(
    nn.Linear(667,300),
    nn.ReLU(),
    nn.Linear(300,100),
    nn.ReLU(),
    nn.Linear(100,20),
    nn.ReLU(),
    nn.Linear(20,1)
)
x = torch.from_numpy(np.load("x.npy"))
y = torch.from_numpy(np.load("y.npy"))

training_loss_list = []
validation_loss_list = []
for i in range(15):
    x_data = x[:,i,:].float()
    y_data = y[:,i,:].float()
    dataset = re_data(x_data,y_data)
    batch_size = 50
    validation_split = 0.2
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    criterion = nn.MSELoss()
    def criterion2(output, target):
        res = torch.abs(output-target)/target
        return res.mean()

    learning_rate = 0.01
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    num_epoch = 10

    #Training loss saved here
    training_loss =[]


    for epoch in range(num_epoch):
        #Training
        for batch_index, (data,target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion2(output, target)
            loss.backward()
            optimizer.step()
            training_loss.append([epoch,round(loss.item(),3)])
    training_loss_list.append([i+1,training_loss])
    loss1_list = []
    loss2_list = []

    with torch.no_grad():
        for batch_index, (data,target) in enumerate(validation_loader):
            output = model(data)
            loss1 = criterion(output,target)
            loss1_list.append(loss1.item())
            loss2 = criterion2(output,target)
            loss2_list.append(loss2.item())
    loss1_avg = np.average(loss1_list)
    loss2_avg = np.average(loss2_list)
    validation_loss_list.append([i+1,round(loss2_avg,3)])
    validation_loss_list.append([i+1,round(loss1_avg,3)])

       
    


#Write the average
f = open("output.txt","w")
f.write('Training Loss: \n')
f.write(" ".join(training_loss_list.__str__()))
#f.write("\nAverage of loss1_list\n")
#f.write(str(loss1_avg))
f.write("\nValidation Loss: loss2, mseloss order\n")
f.write(" ". join(validation_loss_list.__str__()))
f.close()
