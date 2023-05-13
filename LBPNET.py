import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from lbp import lbp_descriptor


# Define LBPNet architecture with feature extraction in dense grid
class LBPNet(nn.Module):
    def __init__(self, num_classes, num_subsets=16):
        super(LBPNet, self).__init__()

        # Define LBP descriptor function and PCA for feature extraction
        self.lbp_descriptor = lbp_descriptor
        self.pca = PCA(n_components=num_subsets)

        # Define the rest of the network architecture
        self.features = nn.Sequential(
            nn.Linear(num_subsets, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Forward pass through dense LBP layers
        desc = self.lbp_descriptor(x)
        self.pca.fit(desc)
        desc = self.pca.transform(desc)

        # Pass features through the network
        x = self.features(torch.tensor(desc).float().cuda())
        x = self.classifier(x)
        return x

