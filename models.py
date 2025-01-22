import torch
import torch.nn as nn

class ASLModel(nn.Module):
    def __init__(self, num_classes=29):
        super(ASLModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        # After 3 pools on a 64x64 input → 8x8 feature map => 128*8*8=8192
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_asl_model(weights_path, device="cpu", num_classes=29):
    """
    Loads the ASLModel with saved weights from weights_path.
    """
    model = ASLModel(num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model