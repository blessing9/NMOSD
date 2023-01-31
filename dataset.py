import numpy as np
import pandas as pd
import torch
import os,sys
from datetime import datetime

os.chdir(sys.path[0])
default_data = {
    "blood": [2.03, 4.389],
    "blood_test": [7.0, 29.0, 62.7],
    "thyroid": [2.82, 1.234, 0.912, 7.35, 2.29, 80.816, 39.46],
    "ana18": [0],
    "aqp4igg": [0],
    "t_b": [236,1580,868],
    "il_6": [2.0],
    "medicine": [0, 0, 0, 0, 0, 1],
    "visit": [],
    "edss": [0.0],
    "vitd": [18.6, 49.88],
    "ig3": [10.39, 1.83, 1.00],
    "wbc": [8.0]
}

# 计算day1到day2的日期差
def dis(day1, day2):
    try:
        time1 = datetime.strptime(day1, '%Y-%m-%d')
        time2 = datetime.strptime(day1, '%Y-%m-%d')
        result = time2.toordinal() - time1.toordinal()
        return result
    except ValueError:
        print(f'date error:{day1},{day2}')
        return 0

class divide_set(torch.utils.data.Dataset):
    def __init__(self, ptr, parent):
        self.ptr = ptr
        self.parent = parent

    def __getitem__(self, item):
        return self.parent.__getitem__(self.ptr[item])

    def __len__(self):
        return len(self.ptr)


def get_train_test(base, i):
    train = []
    test = []
    for j in range(5):
        if i == j:
            test = base.sets[j]
        else:
            train += base.sets[j]
    return divide_set(train, base), divide_set(test, base)

class PointSet(torch.utils.data.Dataset):
    def __init__(self, args):
        self.value_point = pd.read_csv(args.value_point_file).drop(['id','time'], axis=1)
        device = args.device
        self.data = []
        self.pos, self.neg = 0, 0
        for line in self.value_point.iloc:
            line = line.values
            if line[-2] > 365:#将\delta t限制在一年以内
                continue
            x_tensor = torch.tensor(line[:-2], dtype=torch.float).to(device)
            t_tensor = torch.tensor(line[-2], dtype=torch.int).to(device)
            e_tensor = torch.tensor(line[-1], dtype=torch.int).to(device)
            self.data.append((x_tensor, (t_tensor, e_tensor)))
            self.pos += e_tensor.item()
        
        self.data.sort(key=lambda x:x[1][1].item(),reverse=True)
        self.cut_set(args.fold)  #划分交叉验证数据集
        self.input_size = self.data[0][0].shape[0]
        self.neg = len(self.data) - self.pos
        print(f'positive:{self.pos}/negative:{self.neg}')

    def cut_set(self,fold):
        pos = self.pos
        all = self.__len__() - self.pos

        df_set = []
        for i in range(fold):
            divide = []
            for j in range(int((pos / fold) * i), int((pos / fold) * (i + 1))):
                divide.append(j)
            for j in range(int((all / fold) * i) + pos, int((all / fold) * (i + 1)) + pos):
                divide.append(j)
            df_set.append(divide)
        self.sets = df_set

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def pad_zero(k, len_max):
    l = len(k)
    res = k
    for i in range(l, len_max):
        t = np.zeros(res[0].shape)
        res.append(t)
    res = np.vstack(res)
    return res, l

        
class PointSeqSet(torch.utils.data.Dataset):
    def __init__(self, args):
        self.vp = pd.read_csv(args.value_point_file)
        self.seq = open(args.point_seq_file, "r").readlines()
        self.device = args.device
        self.task = args.task #风险预测任务或者二分类任务
        self.len_max = max([(len(i.strip().split(','))-3) for i in self.seq])
        self.init()
        self.cut_set(args.fold)
        self.input_size = self.data[0][0][0].shape[1]
        self.pos = int(sum([i[1][1].item() for i in self.data]) if self.task.__contains__('hazard') else sum([i[1].item() for i in self.data]))
        self.neg = len(self.data)-self.pos
        print(f'positive:{self.pos}/negative:{self.neg}')

    def cut_set(self,fold):
        self.sets = [[] for i in range(fold)]
        for i in range(len(self.data)):
            begin = self.data[i][1][2]
            neg = self.data[i][1][0]
            pos = self.data[i][1][1]
            for j in range(len(neg)):
                self.sets[i%fold].append(begin)
                begin += 1
            for j in range(len(pos)):
                self.sets[i%fold].append(begin)
                begin += 1
        temp = []
        self.ids = []
        for d in self.data:
            neg = d[1][0]
            pos = d[1][1]
            for i in neg:
                temp.append(i)
                self.ids.append(d[0])
            for i in pos:
                temp.append(i)
                self.ids.append(d[0])
        self.data = temp

    def init(self):
        self.data = {} #以病人为单位存储点序列
        for i in range(len(self.seq)):
            line = self.seq[i].strip().split(',')
            id = int(line.pop(0))
            event = int(eval(line.pop(-1)))
            time = int(line.pop(-1))
            #如果进行二分类的实验
            if self.task.__contains__('binary'):
                if event == 0 and time<365:#对于观测时间短于一年的数据，去掉
                    continue
                event = 1 if time < 365 and event==1 else 0
            if event == 0 and time<=365:
                continue
            if event == 1 and time >365:
                continue
            if id not in self.data.keys():
                self.data[id] = [[],[]]
            index = [int(x) for x in line]
            points = self.vp.iloc[index, :].values

            if event == 0 and points.shape[0]<4:
                continue


            breakout = sum([points[i][4] for i in range(points.shape[0])])
            if breakout>2:
                continue
            x = []
            for j in range(points.shape[0]):  #对每一个点
                temp = points[j][2:-2]
                x.append(temp)
            avg = points[0]
            for j in range(1, points.shape[0]):
                avg += points[j]
            avg = np.delete(avg, [0, 1, 4, -2, -1], 0)
            avg /= points.shape[0]

            x, l = pad_zero(x, self.len_max) #填充点序列数据到长度达到len_max
            # constructs torch.Tensor object
            x_tensor = torch.tensor(x.astype(float), dtype=torch.float).to(self.device)
            l_tensor = torch.tensor(l, dtype=torch.int64).to(self.device)
            t_tensor = torch.tensor(time, dtype=torch.float).to(self.device)
            e_tensor = torch.tensor(event, dtype=torch.float).to(self.device)
            avg_tensor = torch.tensor(avg.astype(float), dtype=torch.float).to(self.device)
            if self.task.__contains__('hazard'):
                self.data[id][event].append(((x_tensor, l_tensor, avg_tensor),(t_tensor, e_tensor)))
            else:
                self.data[id][event].append(((x_tensor, l_tensor, avg_tensor), e_tensor))
        self.data = list(self.data.items())
        self.data.sort(key=lambda x:len(x[1][1]),reverse=True)
        order = 0
        for i in range(len(self.data)):
            self.data[i][1].append(order)
            order += len(self.data[i][1][0]) + len(self.data[i][1][1])
        


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class DataSeqSet(torch.utils.data.Dataset):
    def __init__(self, args):
        with open(args.data_seq_file,'r') as f:
            self.data = f.readlines()
        #获取各个检测项目的序列的最长长度
        names = list(eval(self.data[0]).keys())
        names.remove('duration')
        names.remove('id')
        names.remove('event')
        self.len_max = {}
        self.input_size = []
        for n in names:
            self.len_max[n] = max([len(eval(i)[n]) for i in self.data])
            self.input_size.append(max([len(eval(i)[n][0]) for i in self.data if len(eval(i)[n])>0]))
        self.device = args.device
        self.init()
        self.set_ids = [[] for i in range(5)]
        self.cut_set(args.fold)
        for i in range(len(self.set_ids)):
            self.set_ids[i] = list(set(self.set_ids[i]))
    def cut_set(self,fold):
        self.sets = [[] for i in range(fold)]
        for i in range(len(self.samples)):
            begin = self.samples[i][1][2]
            neg = self.samples[i][1][0]
            pos = self.samples[i][1][1]
            for j in range(len(neg)):
                self.sets[i%fold].append(begin)
                self.set_ids[i%fold].append(self.samples[i][0])
                begin += 1
            for j in range(len(pos)):
                self.sets[i%fold].append(begin)
                self.set_ids[i%fold].append(self.samples[i][0])
                begin += 1
        temp = []
        self.ids = []
        for d in self.samples:
            neg = d[1][0]
            pos = d[1][1]
            for i in neg:
                temp.append(i)
                self.ids.append(d[0])
            for i in pos:
                temp.append(i)
                self.ids.append(d[0])
        self.samples = temp
    def init(self):
        self.samples = {}
        self.pos = 0
        len_max = self.len_max
        for i in range(self.__len__()):
            sample = eval(self.data[i])
            id = int(sample['id'])
            event = int(sample['event'])
            self.pos += event
            if id not in self.samples.keys():
                self.samples[id] = [[], []]
            
            info, length = [],[]
            for k in len_max.keys():
                if len(sample[k]) == 0:
                    sample[k].append(default_data[k])
                sample[k] = [np.array(x) for x in sample[k]]
                sample[k], l = pad_zero(sample[k], len_max[k])
                info.append(torch.tensor(sample[k].astype(float),dtype=torch.float).to(self.device))
                length.append(torch.tensor(l, dtype=torch.int64))
            if sample['duration'] > 1000:
                continue
            t_tensor = torch.tensor(sample['duration'],dtype=torch.float).to(self.device)
            e_tensor = torch.tensor(sample['event'], dtype=torch.float).to(self.device)
            self.samples[id][event].append(((info, length),(t_tensor, e_tensor)))
            print(f'{i+1}/{self.__len__()} initialized.', end='\r')
        print()
        self.samples = list(self.samples.items())
        self.samples.sort(key=lambda x:len(x[1][1]),reverse=True)
        order = 0
        for i in range(len(self.samples)):
            self.samples[i][1].append(order)
            order += len(self.samples[i][1][0]) + len(self.samples[i][1][1])
    def __getitem__(self, item):
        return self.samples[item]
    def __len__(self):
        return len(self.data)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        if args.task == 'point_hazard':
            data_set = PointSet(args)
        elif args.task == 'point_seq_hazard' or args.task == 'point_seq_binary':
            data_set = PointSeqSet(args)
        elif args.task == 'data_seq_hazard':
            data_set = DataSeqSet(args)
        else:
            pass
        self.data = data_set.data
        self.input_size = data_set.input_size
        self.sets = data_set.sets
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)