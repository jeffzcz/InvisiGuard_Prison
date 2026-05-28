from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class TimeDistributed(nn.Module):
    """
    Applies a given module to each time step of a sequence independently.
    Input shape:  (batch_size, time_steps, input_dim1, input_dim2, ...)
    Output shape: (batch_size, time_steps, output_dim1, output_dim2, ...)
    """

    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def _flatten_time(self, size):
        """
        Flatten the time dimension into the batch dimension.

        Args:
            size (tuple): Shape of the input tensor, typically (B, T, ...).

        Returns:
            tuple: Flattened size, e.g., (B*T, ...).
        """
        size = list(size)  # Convert to list for manipulation
        return (size[0] * size[1], *size[2:])

    def _restore_time(self, size, batch, time_dim):
        """
        Restore the batch and time dimensions after processing.

        Args:
            size (tuple): Shape after module processing, typically (B*T, ...).
            batch (int): Original batch size.
            time_dim (int): Original time dimension size.

        Returns:
            tuple: Restored shape, e.g., (B, T, ...).
        """
        size = list(size)
        return (batch, time_dim, *size[1:])

    def forward(self, x):
        """
        Forward pass of TimeDistributed.

        Args:
            x (Tensor): Input tensor of shape (B, T, ...).

        Returns:
            Tensor: Output tensor of shape (B, T, ...).
        """
        # Flatten the batch and time dimensions
        x_reshaped = x.contiguous().view(self._flatten_time(x.size()))

        # Apply the wrapped module
        y = self.module(x_reshaped)

        # Restore original batch and time dimensions
        y = y.contiguous().view(self._restore_time(y.size(), x.size(0), x.size(1)))

        return y


class STN3d(nn.Module):
    """
    Input Transform Network (T-Net) to learn a 3x3 affine matrix for input alignment.
    """
    def __init__(self, input_channels=3):
        super(STN3d, self).__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, input_channels * input_channels)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.input_channels).flatten().astype(np.float32))).view(1, -1).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.input_channels, self.input_channels)
        return x


class STNkd(nn.Module):
    """
    Feature Transform Network (T-Net) to learn a kxk matrix for feature alignment.
    """
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    """
    Core PointNet feature extractor. Outputs either global feature vector or point-wise features.
    """
    def __init__(self, input_channels=3, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(input_channels=input_channels)
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size(2)
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        trans_feat = None
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    """
    PointNet classification network.
    """
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    """
    PointNet dense classification (segmentation) network.
    """
    def __init__(self, k=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size(0)
        n_pts = x.size(2)
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    """
    Computes the orthogonality regularization loss for the feature transform matrix.
    """
    d = trans.size(1)
    batchsize = trans.size(0)
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class Sub_PointNet(nn.Module):
    def __init__(self, input_channels=3, feature_transform=False):
        super(Sub_PointNet, self).__init__()
        self.pointnet = PointNetfeat(input_channels=input_channels, global_feat=True, feature_transform=feature_transform)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B*T, N, C) → (B*T, C, N)
        out, _, _ = self.pointnet(x)  # output: (B*T, 1024)
        return out


class SubPointNet(nn.Module):
    """
    Submodule that applies PointNet feature extraction to a single frame of point cloud.
    Input: (B, N, C) where N = number of points, C = channels (e.g., 4 for X, Y, Z, V)
    Output: (B, 1024) global feature vector per frame
    """
    def __init__(self, input_channels=3, feature_transform=False):
        super(SubPointNet, self).__init__()
        self.pointnet = PointNetfeat(input_channels=input_channels, global_feat=True, feature_transform=feature_transform)

    def forward(self, x):
        # Input: (B, N, C) → Transpose to (B, C, N) for PointNet
        x = x.permute(0, 2, 1)
        out, _, _ = self.pointnet(x)
        return out  # Output shape: (B, 1024)


class LSTM_HAR_model(nn.Module):
    def __init__(self, input_dim=600, hidden_dim=64, output_dim=10, num_layers=1, dropout=0.1):
        super(LSTM_HAR_model, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)

        self.fc1 = nn.Linear(hidden_dim * 2, 128)  # bidirectional = hidden_dim * 2
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        # Input x: (batch_size, time_steps=80, features=600)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, time_steps, hidden_dim*2)
        final_feature = lstm_out[:, -1, :]  # Take last time step output
        x = self.fc1(final_feature)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LSTM_HAR_model_2(nn.Module):   # replace mlp with linear layer
    def __init__(self, input_dim=600, hidden_dim=64, output_dim=10, num_layers=1, dropout=0.1):
        super(LSTM_HAR_model_2, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # bidirectional = hidden_dim * 2

    def forward(self, x):
        # Input x: (batch_size, time_steps=80, features=600)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, time_steps, hidden_dim*2)
        final_feature = lstm_out[:, -1, :]  # Take last time step output
        x = self.dropout(final_feature)
        x = self.fc(x)
        return x
    

class MLP_HAR_model(nn.Module):
    """
    MLP model for Human Activity Recognition using flattened point cloud input.
    Architecture:
    - Input: (batch_size, 48000)
    - FC1: 48000 → 64
    - FC2: 64 → 128
    - FC3: 128 → 128
    - FC4: 128 → 64
    - Output: 64 → num_classes
    """

    def __init__(self, input_dim=48000, output_dim=10, dropout_rate=0.1):
        super(MLP_HAR_model, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # Flatten input: (B, 80, 150, 4) → (B, 48000)
        x = x.view(x.size(0), -1)
        return self.model(x)



class PBert_HAR_model(nn.Module):
    def __init__(self, output_dim, frame_num, input_channels=3, dropout_rate=0.1, feature_transform=False):
        super(PBert_HAR_model, self).__init__()
        self.frame_num = frame_num
        self.d_model = 1024

        # (1) TimeDistributed PointNet
        self.pointnet = TimeDistributed(nn.Sequential(
            Sub_PointNet(input_channels=input_channels, feature_transform=feature_transform)
        ))

        # (2) Learnable Positional Encoding: (1, T, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, frame_num, self.d_model))

        # (3) Transformer Encoder: 1 layer, 8 heads, FFN=2048
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout_rate,
            batch_first=True  # (B, T, C)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # (4) Classifier head (mean pooled)
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, output_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, data):
        # Input: (B, T, N, C)
        x = self.pointnet(data)  # → (B, T, 1024)
        x = x + self.pos_embedding  # Add learnable positional encoding

        x = self.transformer(x)  # (B, T, 1024)

        # Mean pooling across time dimension
        x = x.mean(dim=1)  # (B, 1024)

        return self.classifier(x)  # (B, output_dim)


class Pointnet_LSTM_HAR_model(nn.Module):
    def __init__(self, output_dim, frame_num, input_channels=3, dropout_rate=0.1, feature_transform=False):
        super(Pointnet_LSTM_HAR_model, self).__init__()

        self.pointnet = TimeDistributed(nn.Sequential(
            Sub_PointNet(input_channels=input_channels, feature_transform=feature_transform)
        ))

        # Change: Bi-directional LSTM
        self.lstm_net = nn.LSTM(
            input_size=1024,
            hidden_size=32,
            num_layers=1,
            dropout=0,
            bidirectional=True  # <-- bidirectional set to True
        )

        # Change: 32 (hidden) * 2 (bi-directional) = 64
        self.dense = nn.Sequential(
            nn.Linear(frame_num * 64, output_dim),
            nn.Dropout(dropout_rate),
            # nn.Softmax(dim=1)
        )

    def forward(self, data):
        data = self.pointnet(data)  # (B, T, 1024)
        data = data.permute(1, 0, 2)  # (T, B, 1024)
        data, _ = self.lstm_net(data)  # (T, B, 64)
        data = data.permute(1, 0, 2)  # (B, T, 64)
        data = data.reshape(data.size(0), -1)  # (B, T*64)
        return self.dense(data)


class Pointnet_LSTM_HAR_model_2(nn.Module):
    def __init__(self, output_dim, frame_num, input_channels=3, dropout_rate=0.1, feature_transform=False):
        super(Pointnet_LSTM_HAR_model_2, self).__init__()

        self.pointnet = TimeDistributed(nn.Sequential(
            Sub_PointNet(input_channels=input_channels, feature_transform=feature_transform)
        ))

        # Change: Bi-directional LSTM
        self.lstm_net = nn.LSTM(
            input_size=1024,
            hidden_size=32,
            num_layers=1,
            dropout=0,
            bidirectional=True  # <-- bidirectional set to True
        )

        # Change: Two-layer MLP with ReLU activation
        self.dense = nn.Sequential(
            nn.Linear(frame_num * 64, 128),  # First layer
            nn.ReLU(),                       # Activation function
            nn.Dropout(dropout_rate),        # Dropout
            nn.Linear(128, output_dim)       # Second layer
        )

    def forward(self, data):
        data = self.pointnet(data)  # (B, T, 1024)
        data = data.permute(1, 0, 2)  # (T, B, 1024)
        data, _ = self.lstm_net(data)  # (T, B, 64)
        data = data.permute(1, 0, 2)  # (B, T, 64)
        data = data.reshape(data.size(0), -1)  # (B, T*64)
        return self.dense(data)
    




class Pointnet_Maxpool_HAR_model(nn.Module):
    """
    Human Activity Recognition model using PointNet + Temporal Max Pooling.
    Input: (B, T, N, C) → T frames, N points per frame, C channels per point.
    Output: (B, output_dim) → Class scores.
    """
    def __init__(self, output_dim, frame_num, input_channels=3, dropout_rate=0.1, feature_transform=False):
        super(Pointnet_Maxpool_HAR_model, self).__init__()

        self.frame_num = frame_num
        self.pointnet_frame = TimeDistributed(SubPointNet(input_channels=input_channels, feature_transform=feature_transform))

        # MLP classifier after temporal max pooling
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        
            nn.Linear(128, output_dim)
            # nn.Softmax(dim=1)  # optional: if your loss expects probabilities
        )

    def forward(self, x):
        # Input: (B, T, N, C)
        frame_features = self.pointnet_frame(x)  # (B, T, 1024)
        pooled_features = torch.max(frame_features, dim=1).values  # Temporal max pooling: (B, 1024)
        return self.classifier(pooled_features)  # Output: (B, output_dim)


class Pointnet_Maxpool_HAR_model_2(nn.Module):
    """
    Human Activity Recognition model using PointNet + Temporal Max Pooling.
    Input: (B, T, N, C) → T frames, N points per frame, C channels per point.
    Output: (B, output_dim) → Class scores.
    """
    def __init__(self, output_dim, frame_num, input_channels=3, dropout_rate=0.1, feature_transform=False):
        super(Pointnet_Maxpool_HAR_model_2, self).__init__()

        self.frame_num = frame_num
        self.pointnet_frame = TimeDistributed(SubPointNet(input_channels=input_channels, feature_transform=feature_transform))

        # Single linear layer without activation function
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(1024, output_dim)

    def forward(self, x):
        # Input: (B, T, N, C)
        frame_features = self.pointnet_frame(x)  # (B, T, 1024)
        pooled_features = torch.max(frame_features, dim=1).values  # Temporal max pooling: (B, 1024)
        x = self.dropout(pooled_features)
        return self.classifier(x)  # Output: (B, output_dim)