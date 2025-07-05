import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p=0.1)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(p=0.1)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(p=0.1)
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(p=0.1)
        
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.drop5 = nn.Dropout(p=0.1)
        
        self.flattened_size = self._get_flattened_size()
        print("flattened_size:", self.flattened_size)

        self.fc1 =  nn.Linear(self.flattened_size, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc1_drop = nn.Dropout(p=0.4)
        
        self.fc2 =  nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc2_drop = nn.Dropout(p=0.4)
        
        self.fc3 =  nn.Linear(512, 512)
        self.fc3_bn = nn.BatchNorm1d(512)
        self.fc3_drop = nn.Dropout(p=0.2)
        
        self.fc4 =  nn.Linear(512, 136)    
        
    def _get_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 224, 224)
            x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
            x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
            x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
            x = self.drop4(self.pool4(F.relu(self.bn4(self.conv4(x)))))
            x = self.drop5(self.pool5(F.relu(self.bn5(self.conv5(x)))))
            return x.view(1, -1).size(1)
        
    def forward(self, x):
        #print("x.shape:", x.shape)
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool4(F.relu(self.bn4(self.conv4(x)))))
        x = self.drop5(self.pool5(F.relu(self.bn5(self.conv5(x)))))
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1_drop(F.relu(self.fc1_bn(self.fc1(x))))
        x = self.fc2_drop(F.relu(self.fc2_bn(self.fc2(x))))
        #x = self.fc3_drop(F.relu(self.fc3_bn(self.fc3(x))))     
        x = self.fc4(x)     
        
        return x