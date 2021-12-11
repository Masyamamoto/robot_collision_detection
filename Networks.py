import torch
import torch.nn as nn
# from Data_Loaders import Data_Loaders
import numpy

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.input_to_hidden1 = nn.Linear(6,5)
        self.nonlinear_activation1 = nn.ReLU()
        self.input_to_hidden2 = nn.Linear(5,3)
        self.nonlinear_activation2 = nn.ReLU()
        self.input_to_hidden3 = nn.Linear(3,1)
        self.nonlinear_activation3 = nn.Sigmoid()


    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        hidden = self.input_to_hidden1(input)
        hidden = self.nonlinear_activation1(hidden)
        hidden = self.input_to_hidden2(hidden)
        hidden = self.nonlinear_activation2(hidden)
        hidden = self.input_to_hidden3(hidden)
        output = self.nonlinear_activation3(hidden)
        return output


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        predictions, actuals = list(),list()
        loss = 0
        for i, sample in enumerate(test_loader):
            test_input, test_target = sample['input'], sample['label']
            yhat = model(test_input)
            yhat = yhat.detach().numpy()
            predictions.append(yhat)
            actuals.append(test_target.detach().numpy())

        predictions = numpy.vstack(predictions)
        predictions = torch.tensor(predictions)
        predictions = torch.reshape(predictions, [1,len(predictions)])

        actuals = numpy.vstack(actuals)
        actuals = torch.tensor(actuals)
        actuals = torch.reshape(actuals, [1,len(actuals)])
        loss=loss_function(predictions, actuals).tolist()
        return loss

def main():
    # batch_size = 32
    # data_loaders = Data_Loaders(batch_size)
    # test_loader = data_loaders.test_loader
    # loss_function = nn.MSELoss()
    # model = Action_Conditioned_FF()
    # loss = model.evaluate(model,test_loader, loss_function)
    return None

if __name__ == '__main__':
    main()
