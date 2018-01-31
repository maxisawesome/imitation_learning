import torch.nn as nn

class Phi(nn.Module):
    """
    State encoding function discussed in the paper being reproduced
    """
    def __init__(self, input_shape=(1, 80, 80)):
        super(Phi, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 5)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(32, 64, 2)
        self.relu3 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(3, 2)
        self.fc1 = nn.Linear(3136, 64)
        self.relu4 = nn.ReLU()
        
    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)
        output = self.max_pool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.max_pool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        output = self.max_pool3(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.relu4(output)
        #print('output.size():', output.size())
        return output
