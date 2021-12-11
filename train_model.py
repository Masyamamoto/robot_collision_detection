from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle


def train_model(no_epochs):

    batch_size =16
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    loss_function = nn.BCELoss()

    validation_losses = []
    training_losses = []
    epoch_num = []
    # min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    # validation_losses.append(min_loss)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    for epoch_i in range(no_epochs):
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            optimizer.zero_grad()

            input = sample['input']
            target = torch.reshape(sample['label'],[len(sample['label']),1])

            yhat = model(input)
            loss = criterion(yhat, target)
            loss.backward()
            optimizer.step()
        training_losses.append(loss.tolist())
        min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        print(loss.tolist(), min_loss, "{0}% is done!".format(int(epoch_i/no_epochs*100)))
        validation_losses.append(min_loss)
        epoch_num.append(epoch_i+1)
    torch.save(model.state_dict(), 'saved/saved_model.pkl', _use_new_zipfile_serialization=False)
    plt.plot(epoch_num,validation_losses, label='validation_loss')
    plt.plot(epoch_num,training_losses, label='training_loss')
    plt.xlabel('# of epoch of batch_size of {0}'.format(batch_size))
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    no_epochs = 100
    train_model(no_epochs)
