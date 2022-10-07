# import modules and libraries
import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam

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


class BaseModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=3, drop_prob=0.5, lr=1e-4, d_lr=1e-4, batch_size=20, beta1=0.5, beta2=0.9, lambda_rpy=0.03, device='cuda'):
        super(BaseModel, self).__init__()

        # params for configuring the model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop_prob = drop_prob

        # optimizing the model
        self.lr = lr
        self.d_lr = d_lr
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda_rpy = lambda_rpy # based on paper
        self.device = device

    def print_log(self):
        print(self.loss_msg)
        print(self.acc_msg)

    def train(self, encoder_tgt, discriminator, epoch, dataloader):
        self.train()
        
        self.epoch = epoch
        losses = {
            loss: []
            for loss in self.loss_names
        }
        accuracies = {
            accuracy: []
            for accuracy in self.accuracy_names
        }

        for batch_idx, (x_src, y_src, x_tgt, y_tgt, x_rpy, y_rpy) in enumerate(dataloader):
            self.x_src, self.y_src = x_src.to(self.device), y_src.to(self.device)
            self.x_tgt, self.y_tgt = x_tgt.to(self.device), y_tgt.to(self.device)
            self.x_rpy, self.y_rpy = x_rpy.to(self.device), y_rpy.to(self.device)

            self.optimize_parameters()

            for loss in self.loss_names:
                losses[loss].append(getattr(self, 'loss_' + loss).detach().item())
            for accuracy in self.accuracy_names:
                accuracies[accuracy].append(getattr(self, 'accuracy_' + accuracy).detach().item())
        
        self.loss_msg = '[Train][{}] Loss:'.format(epoch)
        for loss in self.loss_names:
            self.loss_msg += ' {} {:.3f}'.format(loss, np.mean(losses[loss]))
        self.acc_msg = '[Train][{}] Accuracy:'.format(epoch)
        for accuracy in self.accuracy_names:
            self.acc_msg += ' {} {:.3f}'.format(accuracy, np.mean(accuracies[accuracy]))
        self.print_log()

    def validate(self, epoch, dataloader):
        self.eval()

        losses = {
            loss: []
            for loss in self.loss_names
        }
        accuracies = {
            accuracy: []
            for accuracy in self.accuracy_names
        }

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(dataloader):
                self.x, self.y = x.to(self.device), y.to(self.device)

                self.forward()

                for loss in self.loss_names:
                    losses[loss].append(getattr(self, 'loss_' + loss).detach().item())
                for accuracy in self.accuracy_names:
                    accuracies[accuracy].append(getattr(self, 'accuracy_' + accuracy).detach().item())

                self.loss_msg = '[Val][{}] Loss:'.format(epoch)
                for loss in self.loss_names:
                    self.loss_msg += ' {} {:.3f}'.format(loss, np.mean(losses[loss]))
                self.acc_msg = '[Val][{}] Accuracy:'.format(epoch)
                for accuracy in self.accuracy_names:
                    self.acc_msg += ' {} {:.3f}'.format(accuracy, np.mean(accuracies[accuracy]))
                self.print_log()

    def test(self, epoch, dataloader):
        self.eval()

        losses = {
            loss: []
            for loss in self.loss_names
        }
        accuracies = {
            accuracy: []
            for accuracy in self.accuracy_names
        }

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(dataloader):
                self.x, self.y = x.to(self.device), y.to(self.device)

                self.forward()

                for loss in self.loss_names:
                    losses[loss].append(getattr(self, 'loss_' + loss).detach().item())
                for accuracy in self.accuracy_names:
                    accuracies[accuracy].append(getattr(self, 'accuracy_' + accuracy).detach().item())

                self.loss_msg = '[Test][{}] Loss:'.format(epoch)
                for loss in self.loss_names:
                    self.loss_msg += ' {} {:.3f}'.format(loss, np.mean(losses[loss]))
                self.acc_msg = '[Test][{}] Accuracy:'.format(epoch)
                for accuracy in self.accuracy_names:
                    self.acc_msg += ' {} {:.3f}'.format(accuracy, np.mean(accuracies[accuracy]))
                self.print_log()

class SourceOnly(BaseModel):
    def __init__(self):
        super(SourceOnly, self).__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            self.encoder_src.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2)
            )
        self.loss_names = ['e_pred']

    def forward(self):
        self.e_src = self.encoder_src(self.x_src)
        self.pred_src = self.classifier(self.e_src)
    
    def backward_e(self):
        self.loss_e_pred = self.criterion(self.pred_src, self.y_src)
        self.loss_e = self.loss_e_pred
        self.loss_e.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward_e()
        self.optimizer.step()
        