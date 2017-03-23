import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 x 32 x 32
        self.conv1 = nn.Conv2d(3, 64, 3)
        # 64 x 30 x 30
        self.bn1 = nn.BatchNorm2d(64)
        self.drop1 = nn.Dropout2d(p=0.3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        # 64 x 28 x 28
        self.bn2 = nn.BatchNorm2d(64)
        # 64 x 14 x 14
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # 128 x 14 x 14
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout2d(p=0.4)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.drop4 = nn.Dropout2d(p=0.4)
        # 128 x 14 x 14
        # 128 x 7 x 7
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        # 256 x 7 x 7
        self.bn5 = nn.BatchNorm2d(256)
        self.drop5 = nn.Dropout2d(p=0.4)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        # 256 x 7 x 7
        self.bn6 = nn.BatchNorm2d(256)
        self.drop6 = nn.Dropout2d(p=0.4)
        self.conv7 = nn.Conv2d(256, 256, 3)
        # 256 x 6 x 6
        self.bn7 = nn.BatchNorm2d(256)
        self.drop7 = nn.Dropout2d(p=0.4)
        # 256 x 3 x 3
        self.conv8 = nn.Conv2d(256, 512, 3)
        self.bn8 = nn.BatchNorm2d(512)
        self.drop8 = nn.Dropout2d(p=0.4)

        self.fc9 = nn.Linear(512, 512)
        self.bn9 = nn.BatchNorm1d(512)
        self.drop9 = nn.Dropout()

        self.fc10 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = F.max_pool2dF.relu((self.bn2(self.conv2(x)), (2,2)))
        x = self.drop3(F.relu(self.bn3(self.conv3(x))))
        x = F.max_pool2d(self.drop4(F.relu(self.bn4(self.conv4(x))), (2,2)))
        x = self.drop5(F.relu(self.bn5(self.conv5(x))))
        x = self.drop6(F.relu(self.bn6(self.conv6(x))))
        x = F.max_pool2d(self.drop7(F.relu(self.bn7(self.conv7(x))), (2,2)))
        x = self.drop8(F.relu(self.bn8(self.conv8(x))))
        x = x.view(-1, 512)
        x = self.drop9(F.relu(self.bn9(self.fc9(x))))
        x = self.fc10(x)
        return x
