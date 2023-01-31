from torch import nn
import torch
import numpy as np


#point-based hazard predicting model
class PointHazard(nn.Module):
    def __init__(self, args):
        super().__init__()
        layers = []
        fc = [args.input_size] + args.fc_point
        for i in range(len(fc) - 1):
            layers.append(nn.Linear(fc[i],fc[i + 1]))
            layers.append(nn.BatchNorm1d(fc[i + 1]))
            layers.append(nn.SELU())
        self.layer = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.layer(x)
        x = x.squeeze(-1)
        return x

#point seq-based hazard predictiing model
class PointSeqHazard(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.gru = nn.GRU(args.input_size, args.hidden_size_point_seq, num_layers=1, batch_first=True)  # utilize the LSTM model in torch.nn
        layers = []
        fc = [args.hidden_size_point_seq] + args.fc_point_seq
        for i in range(len(fc) - 1):
            layers.append(nn.Dropout(0.1))
            layers.append(nn.Linear(fc[i], fc[i + 1]))
            layers.append(nn.BatchNorm1d(fc[i + 1]))
            layers.append(nn.SELU())
        self.layer = nn.Sequential(*layers)

        self.avg_fc = nn.Linear(args.input_size-1, args.hidden_size_point_seq)

    def forward(self, x):
        input, lengths, avg = x
        data = nn.utils.rnn.pack_padded_sequence(input, lengths.cpu(), self.gru.batch_first, enforce_sorted=False)
        h0 = self.avg_fc(avg).unsqueeze(0)
        _, hn = self.gru(data, h0)
        s,b,h = hn.shape
        x = hn.view(s*b, h)
        # x = torch.cat((x,avg),1)
        x = self.layer(x)
        x = x.squeeze(-1)
        return x


class PointSeqBinary(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.gru = nn.GRU(args.input_size, args.hidden_size_point_seq, num_layers=1,batch_first=True)
        layers = []
        fc = [args.hidden_size_point_seq] + args.fc_point_seq
        for i in range(len(fc) - 1):
            layers.append(nn.Dropout(0.1))
            layers.append(nn.Linear(fc[i], fc[i + 1]))
            layers.append(nn.BatchNorm1d(fc[i + 1]))
            layers.append(nn.SELU())
        self.layer = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

        self.avg_fc = nn.Linear(args.input_size-1, args.hidden_size_point_seq)
    def forward(self, x):
        input, lengths, avg = x
        data = nn.utils.rnn.pack_padded_sequence(input, lengths.cpu(), self.gru.batch_first, enforce_sorted=False)
        h0 = self.avg_fc(avg).unsqueeze(0)
        _, hn = self.gru(data, h0)
        s, b, h = hn.shape
        x = hn.view(s * b, h)
        x = self.layer(x)
        x = x.squeeze(-1)
        x = self.sigmoid(x)
        return x


#data seq-based hazard predicting model
class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, input, lengths):
        data = nn.utils.rnn.pack_padded_sequence(input, lengths.cpu(), self.lstm.batch_first, enforce_sorted=False)
        result, (hn, cn) = self.lstm(data)
        s, b, h = hn.shape
        x = hn.view(s * b, h)
        x = x.squeeze(-1)
        return x


class DataSeqHazard(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.t = len(args.input_size)           #要用到多少类检测记录
        device = args.device
        self.first_part = []
        for i in range(self.t):
            self.first_part.append(myLSTM(args.input_size[i], args.hidden_size_data_seq[i]).to(device))
        input = np.sum(args.hidden_size_data_seq)        #性别&年龄+九类检测项目通过lstm之后的总维数
        layers = []
        fc = [input] + args.fc_data_seq
        for i in range(len(fc) - 1):
            layers.append(nn.Linear(fc[i], fc[i + 1]))
            layers.append(nn.BatchNorm1d(fc[i + 1]))
            layers.append(nn.SELU())
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        input, lengths = x
        temp = []
        for i in range(self.t):
            temp.append(self.first_part[i](input[i], lengths[i]))     #每一类检测项目经过LSTM后的隐向量
        x = torch.cat(temp,1)
        x = self.layer(x)
        x = x.squeeze(-1)
        return x





class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.task == 'point_hazard':
            self.model = PointHazard(args)
        elif args.task == 'point_seq_hazard':
            self.model = PointSeqHazard(args)
        elif args.task == 'point_seq_binary':
            self.model = PointSeqBinary(args)
        elif args.task == 'data_seq_hazard':
            self.model = DataSeqHazard(args)
        else:
            pass
    def forward(self, x):
        return self.model(x)