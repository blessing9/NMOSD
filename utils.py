from datetime import datetime
from functools import cmp_to_key
from lifelines.utils import concordance_index
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
import numpy as np
import torch
from dataset import get_train_test
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix
import matplotlib.pyplot as plt

default_data = {
    "blood": [7.0, 2.03, 4.389],
    "blood_test": [7.0, 29.0, 62.7],
    "thyroid": [2.82, 1.234, 0.912, 7.35, 2.29, 80.816, 39.46],
    "ana18": [0],
    "aqp4igg": [0],
    "t_b": [236, 11.05, 215, 10.9, 1580, 77.91, 868, 622, 31.1, 42.2],
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
        time2 = datetime.strptime(day2, '%Y-%m-%d')
        result = time2.toordinal() - time1.toordinal()
        return result
    except ValueError:
        print(f'date error:{day1},{day2}')
        return 0

# 从清洗过的file文件中，根据病人id取出所有对应的检测记录,返回列表
def get_from(file, id):
    res = []
    f = open("data/washed_sheet/" + file, "r")
    for i in f.readlines():
        line = i.strip()
        line = line.split(",")
        if int(line[0]) == id:
            temp = [line[1]]
            if file == 'blood.csv':
                temp += line[3:]
            elif file == 'thyroid.csv':
                temp += [line[3], line[5], line[7], line[8]]
            elif file == 'aqp4igg.csv':
                temp += line[2:]
            elif file == 'vitd':
                temp += line[2:]
            elif file == 't_b.csv':
                temp += [line[2],line[6],line[8]]
            elif file == 'ig3.csv':
                temp += line[2:]
            else:
                temp += line[2:]
            res.append(temp)
    res.sort(key=cmp_to_key(lambda a, b: dis(a[0], b[0])), reverse=True)#时间正序
    return res

#判断time是否在points中任何点的{range}天以内(time,points中的元素全部都是表示日期的字符串)
def aroundpoint(points, time, range):
    count = 0
    for p in points:
        if (abs(dis(time,p)))<= range:
            count += 1
    if count > 0:
        return True
    else:
        return False

def nearest_record(records, time, range):
    '''
    寻找records中，离time最近的记录，如果这条记录在time {range}天内，返回这条记录，如果不在，返回-1
    records:记录组成的列表，每一个元素都是代表一条记录的列表
    time:代表日期的字符串
    range:代表搜寻的范围
    '''
    if len(records) == 0:
        return -1
    diff = [abs(dis(time,r[0])) for r in records]
    min_index = len(diff) - diff[::-1].index(min(diff)) - 1#如果两条记录距time一样，那么返回后一条
    if abs(dis(records[min_index][0],time)) <= range:
        return records[min_index]
    else:
        return -1

# 根据用药记录列表l，返回病人在start到end期间所使用的药物（若有多种，返回覆盖区间最长的一种）。
def get_medicine(l, start, end, default):
    cover = 0
    r = default
    for i in range(len(l)):
        s = dis(l[i][0], start)
        if i < len(l) - 1:
            e = dis(end, l[i + 1][0])
        else:
            e = 0
        if s >= 0 and e >= 0:
            r = l[i][1:]
        else:
            if s < 0 and e >= 0:
                c = dis(l[i][0], end)
            elif s >= 0 and e < 0:
                c = dis(start, l[i + 1][0])
            else:
                c = dis(l[i][0],l[i+1][0])#错误 从c = dis(start,end)改为 c = dis(l[i][0],l[i+1][0])
            if c > cover:
                cover = c
                r = l[i][1:]
    return r



#跟训练模型相关的功能函数

#计算c-index
def c_index(pred, y):
    t, e = y
    if isinstance(pred, torch.Tensor):
        pred = pred.tolist()
    if isinstance(t, torch.Tensor):
        t = t.tolist()
    if isinstance(e, torch.Tensor):
        e = e.tolist()
    return concordance_index(np.array(t) ,-np.array(pred), np.array(e))



#deepsurv中的negative partial likelihood
def neg_loss(r, y):
    t, e = y
    mask = torch.ones(t.shape[0],t.shape[0]).to(r.device)
    r = torch.where(r<80, r, torch.full_like(r,80))
    time = t * mask - t.view(-1,1)

    log_loss = torch.exp(r) * mask
    log_loss = torch.where(time >= 0, log_loss, torch.zeros_like(log_loss))
    log_loss = torch.sum(log_loss, dim = 1)
    log_loss = torch.log(log_loss).reshape(-1)
    log_loss = torch.sum((r-log_loss) * e)/torch.sum(e)
    return -log_loss

def cross_valid(data_set, paras, train_func):
    for i in range(10):
        train_set, valid_set, test_set = get_train_test(data_set, i)
        train_func(paras, train_set, valid_set, test_set, i)


def PlotConfusionMatrix(y_true, y_pred, order,time):
    temp = y_pred.cpu().detach().numpy()
    y_pred = np.array([1 if i >= 0.5 else 0 for i in temp])
    y_true = y_true.cpu().detach().numpy()
    C = confusion_matrix(y_true,y_pred)
    plt.matshow(C, cmap = plt.cm.Blues)
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i],xy=(i,j), horizontalalignment='center',verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'result/confusion_matrix/2step_binary_{time}year/{order+1}.jpg')
    plt.clf()

def auc(pred, truth, curve=False, np=False):
    # try:
    #     auc = roc_auc_score(truth, pred)
    # except ValueError:
    #     return 0.5
    # return auc
    if isinstance(pred, torch.Tensor):
        pred = pred.tolist()
    if isinstance(truth, torch.Tensor):
        truth = truth.tolist()
    return roc_auc_score(truth, pred)