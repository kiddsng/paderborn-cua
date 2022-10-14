# import modules and libraries
import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

### models ###


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=64,
                      stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=16, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=8, stride=2, padding=1, dilation=1),
            nn.Flatten(),
            # change from 3992 to 5016 because of RuntimeError: mat1 and mat2 shapes cannot be multiplied (20x5016 and 3992x64)
            nn.Linear(5016, self.hidden_dim)
        )

    def forward(self, x):
        # reshape input to (batch_size, input_dim (1), sequence length (5120))
        x = x.view(x.size(0), self.input_dim, -1)
        # extract features from sensor readings
        features = self.encoder(x.float())
        # x.float() because of RuntimeError: expected scalar type Double but found Float
        return features


class Classifier(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout):
        super(Classifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )

    def forward(self, x):
        # input: features extracted from encoder
        predictions = self.classifier(x)  # predict from features
        return predictions


class Encoder_AMDA(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder_AMDA, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=32,
                      stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=16, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=8, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=8, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(256, self.hidden_dim)
        )

    def forward(self, x):
        # reshape input to (batch_size, input_dim (1), sequence length (5120))
        x = x.view(x.size(0), self.input_dim, -1)
        # extract features from sensor readings
        features = self.encoder(x.float())
        # x.float() because of RuntimeError: expected scalar type Double but found Float
        return features


class Classifier_AMDA(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout):
        super(Classifier_AMDA, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, x):
        # input: features extracted from encoder
        predictions = self.classifier(x)  # predict from features
        return predictions


# Discriminator

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        output = self.layer(input)

        return output

# https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
def generate_confusion_matrix(encoder, classifier, dataloader, device):
    encoder.eval()
    classifier.eval()

    y_pred = []
    y_true = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = classifier(encoder(x))

        pred = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()
        y_pred.extend(pred)

        labels = y.data.cpu().numpy()
        y_true.extend(labels)

    classes = ('Healthy', 'Inner-bearing Damage', 'Outer-bearing Damage')

    cf_matrix = confusion_matrix(y_true, y_pred)
    num_labels = np.reshape(np.unique(y_true, return_counts=True)[1], (3, 1))
    df_cm = pd.DataFrame(cf_matrix / num_labels, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, cmap="Blues", annot=True)

