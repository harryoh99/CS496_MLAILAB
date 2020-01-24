import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

# os.chdir("/Users/jiooh/Documents/Jio/KAIST/2-2/개별연구/Data_Calibration")


class REData(Dataset):
    def __init__(self, x_data, y_data):
        indice = y_data.nonzero()[:, 0]
        self.data = x_data[indice]
        self.labels = y_data[indice]
        self.len = self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.len


def save_model():
    torch.save(model2, "crt2_model.pt")
    torch.save(model_dropout, "crt2_model_dropout.pt")


def load_model():
    model2 = torch.load("crt2_model.pt")
    model_dropout = torch.load("crt2_model_dropout.pt")
    return model2, model_dropout


# 63560, 15, 667 for x
# 63560, 15, 1 for y
# For year 1


def criterion2(output, target):
    res = torch.abs(output - target) / target
    return res.mean()


def main(xpath, ypath):
    # Model
    model2 = nn.Sequential(
        nn.Linear(667, 300),
        nn.ReLU(),
        nn.Linear(300, 100),
        nn.ReLU(),
        nn.Linear(100, 20),
        nn.ReLU(),
        nn.Linear(20, 1),
    )

    model_dropout = nn.Sequential(
        nn.Linear(667, 300),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(300, 100),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(100, 20),
        nn.ReLU(),
        nn.Linear(20, 1),
    )

    # DATA LOAD/SPLIT
    x = torch.from_numpy(np.load(xpath))
    y = torch.from_numpy(np.load(ypath))

    x_data = x[:, 0, :].float()
    y_data = y[:, 0, :].float()
    for i in range(1, 15):
        temp_x = x[:, i, :].float()
        temp_y = y[:, i, :].float()
        x_data = torch.cat((x_data, temp_x), dim=0)
        y_data = torch.cat((y_data, temp_y), dim=0)

    dataset_size = x_data.shape[0]  # (63560*15,667)
    validation_split = 0.01

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_x = x_data[train_indices]
    train_y = y_data[train_indices]

    test_x = x_data[val_indices]
    test_y = y_data[val_indices]

    train_dataset = REData(train_x, train_y)
    test_dataset = REData(test_x, test_y)
    batch_size = 50

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True)

    learning_rate = 0.01
    optimizer_dropout = optim.Adam(model_dropout.parameters(), lr=learning_rate)
    optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)
    num_epoch = 50

    # Training loss saved here

    training2_loss = []
    training2_epoch_loss = []

    training2_dropout_loss = []
    training2_dropout_epoch_loss = []

    print("dropout model")
    for epoch in range(num_epoch):
        print(epoch)
        # Training
        for batch_index, (data, target) in enumerate(train_loader):
            optimizer_dropout.zero_grad()
            output = model_dropout(data)
            loss = criterion2(output, target)
            loss.backward()
            optimizer_dropout.step()
            training2_dropout_loss.append([epoch, round(loss.item(), 3)])
        training2_dropout_epoch_loss.append(round(loss.item(), 3))
        print(epoch, loss)

    print("regular mode")
    for epoch in range(num_epoch):
        # Training
        for batch_index, (data, target) in enumerate(train_loader):
            optimizer2.zero_grad()
            output = model2(data)
            loss = criterion2(output, target)
            loss.backward()
            optimizer2.step()
            training2_loss.append([epoch, round(loss.item(), 3)])
        training2_epoch_loss.append(round(loss.item(), 3))
        print(epoch, loss)

    save_model()
    model2, model_dropout = load_model()

    # Tested with criterion 2
    loss_list = []
    loss_dlist = []

    with torch.no_grad():
        for batch_index, (data, target) in enumerate(validation_loader):
            output2 = model2(data)  # Trained with criterion 2
            output_dropout = model_dropout(data)
            loss22 = criterion2(
                output2, target
            )  # Trainde with criterion 2, tested with 2
            loss_dropout = criterion2(output_dropout, target)
            loss_list.append(loss22.item())
            loss_dlist.append(loss_dropout.item())

    loss_avg = np.average(loss_list)
    loss_dropout_avg = np.average(loss_dlist)

    # Write the average
    f = open("output.txt", "w")
    f.write("\nTraining Loss with criterion 2: \n")
    f.write(" ".join(training2_loss.__str__()))
    f.write("\nTraining Loss with criterion 2 with dropout: \n")
    f.write(" ".join(training2_dropout_loss.__str__()))
    f.write("\nAverage loss of model trained with criterion 2\n")
    f.write(str(loss_avg))
    f.write("\nAverage loss of model trained with criterion 2 with dropout\n")
    f.write(str(loss_dropout_avg))
    f.close()

    writer = SummaryWriter("./logs/check")
    for idx in range(len(loss_list)):
        writer.add_scalars(
            "test result",
            {"without dropout:": loss_list[idx], "with dropout: ": loss_dlist[idx]},
            idx,
        )

    for idx in range(len(training2_epoch_loss)):
        writer.add_scalars(
            "training result for each epoch",
            {
                "without dropout": training2_epoch_loss[idx],
                "with dropout": training2_dropout_epoch_loss[idx],
            },
            idx,
        )

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--xpath", type=str, default="x.npy", help="path to x file")
    parser.add_argument("--ypath", type=str, default="y.npy", help="path to y file")

    args = parser.parse_args()

    main(args.xpath, args.ypath)
