import csv
from filecmp import cmp
import os,sys
from time import time

import pandas as pd
from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

os.chdir(sys.path[0])



breakout_f = "breakout.csv"
blood_f = "blood.csv"
thyroid_f = "thyroid.csv"
ana18_f = "ana18.csv"
aqp4igg_f = "aqp4igg.csv"
t_b_f = "t_b.csv"
il_6_f = "il_6.csv"
medicine_f = "medicine.csv"
visit_f = "visit.csv"
edss_f = "edss.csv"
vitd_f = "vitd.csv"
ig3_f = "ig3.csv"
ocb_f = "ocb.csv"

used_items = {'blood': [2,3], 't_b': [1], 'vitd': [1,2], 'aqp4igg': [1]}#要使用的检测变量


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



class patient:
    def __init__(self, info):
        h = info.split(",")
        self.id = int(h[0])
        self.birthday = h[1]
        self.gender = int(h[2])
        self.data = [int(x) for x in h[3:]]
        self.init_records()
    def init_records(self):
        self.visit = get_from(visit_f, self.id)
        self.breakout = get_from(breakout_f, self.id)
        self.medicine = get_from(medicine_f, self.id)
        self.records = {
            # "edss": get_from(edss_f, self.id),
            "blood": get_from(blood_f, self.id),
            # "thyroid": get_from(thyroid_f, self.id),
            # "ana18": get_from(ana18_f, self.id),
            "aqp4igg": get_from(aqp4igg_f, self.id),
            "vitd": get_from(vitd_f, self.id),
            "t_b": get_from(t_b_f, self.id),
            "ig3": get_from(ig3_f, self.id),
            # "il_6": get_from(il_6_f, self.id),
        }
        self.ocb = self.get_ocb(ocb_f)
        self.first_breakout = self.breakout[0]   #首次发作记录
    
    #获取每个病人唯一的ocb数据/缺失标为-1
    def get_ocb(self, ocb_f):
        f = open('data/washed_sheet/' + ocb_f,'r')
        ocb = []
        for i in f.readlines():
            line = i.strip()
            line = line.split(',')
            if int(line[0]) == self.id:
                ocb.append(line[1:])
        ocb.sort(key=cmp_to_key(lambda a,b:dis(a[0],b[0])),reverse=True)
        return int(ocb[-1][1]) if len(ocb)>0 else -1

    #记录病人的检测记录数量
    def record_num(self):
        return sum([len(v) for v in self.records.values()])


    #返回time时间点周围range天内的检测项目种类数
    def record_kind_num(self, time, range):
        count = 0
        for v in self.records.values():
            for r in v:
                if abs(dis(r[0], time)) <= range:
                    count += 1
                    break
        return count
    
    #返回病人的第一条检测记录时间，若病人没有任何记录，则返回None
    def first_record(self):
        earliest = []
        for v in self.records.values():
            if len(v) > 0:
                earliest.append(v[0][0])
        if len(earliest) == 0:
            return None
        earliest.sort(key=cmp_to_key(lambda a,b:dis(a,b)),reverse=True)
        return earliest[0]
    
    #返回病人的最后一条检测记录时间，若病人没有任何记录，则返回None
    def last_record(self):
        last = []
        for v in self.records.values():
            if len(v) > 0:
                last.append(v[-1][0])
        if len(self.visit) > 0:
            last.append(self.visit[-1][0])
        if len(self.breakout) > 0:
            last.append(self.breakout[-1][0])
        if len(self.medicine) > 0:
            last.append(self.medicine[-1][0])
        if len(last) == 0:
            return None
        last.sort(key=cmp_to_key(lambda a,b:dis(a,b)),reverse=True)
        return last[-1]

    #剔除掉第一条检测记录前range天之前的所有发作点和访视点
    def filter_useless_points(self,range):
        if self.record_num() == 0:
            self.visit = []
            self.breakout = []
            return
        earliest = self.first_record()
        i = 0
        for r in self.breakout:
            if dis(r[0],earliest)<=range:
                break
            i += 1
        if i == len(self.breakout):
            self.breakout = []
        else:
            self.breakout = self.breakout[i:]

        i = 0
        for r in self.visit:
            if dis(r[0],earliest)<=range:
                break
            i += 1
        if i == len(self.visit):
            self.visit = []
        else:
            self.visit = self.visit[i:]
        return

    #构造有价值点
    def make_point(self,*args):
        time = args[0]
        age = dis(self.birthday, time)//365
        state = 1 if time in [t[0] for t in self.breakout] else 0
        result = [self.id, time, self.gender, age, state] #id,时间，性别，年龄，点本身状态
        
        miss_info = {} #2022.12.7 对于每个有价值点，记录检测项目的缺失信息

        if len(args) == 1:#从发作点和访视点中寻找有价值点
            #获取各项检测记录
            for k,v in self.records.items():
                nearest = nearest_record(v, time, self.time_threshold)
                if nearest == -1:#time周围没有这项纪录，填补缺省值或者-1（对于相关性分析）
                    miss_info[k] = False
                    if self.corr:
                        result += [-1 for i in default_data[k]]
                    else:
                        result += default_data[k]
                else:
                    miss_info[k] = True
                    result += nearest[1:]
        else:#从未被使用的检测记录中提取有价值点
            series = args[1]
            #获取各项纪录
            if len(series) == 1:
                s = series[0][1]
                for k in self.records.keys():
                    if k in s.keys():
                        miss_info[k] = True
                        result += s[k]
                    else:
                        miss_info[k] = False
                        if self.corr:
                            result += [-1 for i in default_data[k]]
                        else:
                            result += default_data[k]

            else:
                temp = {}
                for s in series:
                    for k,v in s[1].items():
                        if k not in temp.keys():
                            temp[k] = []
                        temp[k].append([s[0]] + v)
                for k in self.records.keys():
                    if k in temp.keys():
                        nearest = nearest_record(temp[k],time, self.time_threshold)
                        if nearest == -1:
                            miss_info[k] = False
                            if self.corr:
                                result += [-1 for i in default_data[k]]
                            else:
                                result += default_data[k]
                        else:
                            miss_info[k] = True
                            result += nearest[1:]
                    else:
                        miss_info[k] = False
                        if self.corr:
                            result += [-1 for i in default_data[k]]
                        else:
                            result += default_data[k]
        
        #寻找尾时间点
        latest = {107083039:'2016-08-11',107082629:'2018-05-27',107083207:'2016-01-01',276880426:'2018-08-01',273741566:'2020-04-30',718496886:'2020-06-01'}
        tail = ''
        for t in self.breakout:
            if dis(time,t[0]) > 0:
                tail = t[0]
                break
        if tail == '':
            if self.id in latest.keys():
                tail = latest[self.id]
            else:
                tail = '2021-12-01'

        #获取用药信息
        # result += get_medicine(self.medicine,time,tail,default_data['medicine'])

        # result.append(dis(time, tail))#距尾时间点的时间
        event = 1 if tail in [t[0] for t in self.breakout] else 0 #尾时间点状态
        #距尾时间点的时间/距下次复发的时间
        if self.corr:
            interval = dis(time, tail) if event==1 else -1
        else:
            interval = dis(time, tail)
        result.append(interval)


        if self.corr:
            if dis(time,tail)<self.time * 365 and event == 0:
                result.append(-1)
            else:
                result.append(1 if dis(time,tail)<self.time * 365 and event == 1 else 0)
        else:
            result.append(event)

        # result.append(miss_info)
        valid = miss_info['t_b']
        return result, valid

    #构造所有有价值点，存储在self.value_points里面
    def make_value_points(self, item_threshold,time_threshold,corr,time):
        self.corr = corr #是否进行相关性分析
        self.time = time
        self.time_threshold = time_threshold

        self.value_points = []
        #剔除第1条记录前{time_threshold}天之前的记录
        self.filter_useless_points(time_threshold)

        #剔除周围{time_threshold}天检测记录<3的访视点以及在发作点周围{time_threshold}天的访视点
        breakout = [i[0] for i in self.breakout]
        temp = []
        for t in self.visit:
            if not aroundpoint(breakout, t[0], time_threshold) and self.record_kind_num(t[0],time_threshold) >= item_threshold:
                temp.append(t)
        self.visit = temp


        #提取不在任何访视点和发作点周围的检测项目，并从中提取有价值点
        points = [i[0] for i in self.breakout] + [i[0] for i in self.visit] #所有访视点和发作点
        points.sort(key=cmp_to_key(lambda a,b:dis(a,b)),reverse=True)

        
        unused_records = {}  #以时间为key存储不在任何点周围30天内的检测记录
        for k,v in self.records.items():
            for t in v:
                if not aroundpoint(points,t[0],time_threshold):
                    if t[0] not in unused_records.keys():
                        unused_records[t[0]] = {}
                    unused_records[t[0]][k] = t[1:]
        unused_records = list(unused_records.items())
        unused_records.sort(key=cmp_to_key(lambda a,b:dis(a[0],b[0])),reverse=True)

        series = [] #存储unused_records中元素的序列，如果相隔{time_threshold}天以内就组织到一起
        temp = []
        for i in range(len(unused_records)):
            if i == 0 or dis(unused_records[i-1][0],unused_records[i][0]) <= time_threshold:
                temp.append(unused_records[i])
            else:
                series.append(temp)
                temp = [unused_records[i]]
        if temp:
            series.append(temp)
        for s in series:
            if len(s) == 1:
                if len(s[0][1]) >= item_threshold:
                    #TO DO 添加有价值点
                    ret, valid = self.make_point(s[0][0],s)
                    if valid:
                        self.value_points.append(ret)
                    # self.value_points.append(self.make_point(s[0][0], s))
            else:#选择合并时间点
                times = [dis(s[0][0],i[0]) for i in s]#相对时间
                weights = [len(i[1]) for i in s]
                avg_time = sum([times[i]*weights[i] for i in range(len(times))])/sum(weights)
                diff = [abs(t-avg_time) for t in times]
                vp_index = len(diff) - diff[::-1].index(min(diff)) -1 #合并时间点在s中的索引
                items = []
                for t in s:
                    if abs(dis(t[0], s[vp_index][0])) <= time_threshold:
                        for i in t[1].keys():
                            items.append(i)
                if len(list(set(items))) >= item_threshold:
                    #TO DO 添加有价值点
                    ret, valid = self.make_point(s[vp_index][0],s)
                    if valid:
                        self.value_points.append(ret)
                    # self.value_points.append(self.make_point(s[vp_index][0], s))
        
        #从所有访视点中提取有价值点
        visit = [i[0] for i in self.visit]

        series = []
        temp = []
        for i in range(len(visit)):
            if i == 0 or dis(visit[i-1], visit[i]) <= time_threshold:
                temp.append(visit[i])
            else:
                series.append(temp)
                temp = [visit[i]]
        if temp:
            series.append(temp)
        for s in series:
            if len(s) == 1:
                #TO DO 添加有价值点
                ret, valid = self.make_point(s[0])
                if valid:
                    self.value_points.append(ret)
                # self.value_points.append(self.make_point(s[0]))
            else:
                times = [dis(s[0], t) for t in s]#相对时间
                weights = []#权重
                for t in s:
                    temp = 0
                    for v in self.records.values():
                        nearest = nearest_record(v,t,time_threshold)
                        if nearest != -1:
                            temp += time_threshold - abs(dis(nearest[0],t))
                    weights.append(temp)
                avg_time = sum([times[i]*weights[i] for i in range(len(times))])/sum(weights)
                diff = [abs(t - avg_time) for t in times]
                vp_index = len(diff) - diff[::-1].index(min(diff)) - 1
                #TO DO 添加有价值点
                ret, valid = self.make_point(s[vp_index])
                if valid:
                    self.value_points.append(ret)
                # self.value_points.append(self.make_point(s[vp_index]))
        #给有价值点排序
        self.value_points.sort(key=cmp_to_key(lambda a,b:dis(a[1],b[1])),reverse=True)


        #删掉无用的发作点,将剩下的发作点全部添加至有价值点
        earliest = ''
        for t in self.breakout:
            if self.record_kind_num(t[0],time_threshold) >= item_threshold:
                earliest = t[0]
                break
        if earliest != '':
            if len(self.value_points) > 0:
                earliest = earliest if dis(earliest,self.value_points[0][1])>=0 else self.value_points[0][1]
        else:
            if len(self.value_points) > 0:
                earliest = self.value_points[0][1]
        if earliest != '':
            temp = []
            for t in self.breakout:
                if dis(t[0], earliest) <= 0:
                    ret, valid = self.make_point(t[0])
                    if valid:
                        self.value_points.append(ret)
                    # self.value_points.append(self.make_point(t[0]))
                    temp.append(t)
            self.breakout = temp
        else:
            self.breakout = []

        self.value_points.sort(key=cmp_to_key(lambda a,b:dis(a[1],b[1])),reverse=True)




        
#构造有价值点，存储在value_points.csv中
def make_point_data(patients, item_threshold=3, time_threshold = 30, corr=False, time = 1):
    """
    构造有价值点，存储在value_points.csv中

    参数
    -------
    item_threshold:int
        构造有价值点时，周围检测项目数量的最小阈值
    time_threshold:int
        构造有价值点时，提取检测项目时间范围的最大阈值，默认30天
    corr:bool
        是否进行相关性分析，若进行相关性分析，则有价值点缺少的项目不填缺省值，填-1
    time:int
        计算相关性时用到，计算各检测项目与{time}年内复发的相关性
    """
    if not corr:
        w = csv.writer(open(f'data/value_points.csv', 'w', newline=''))
    else:
        w = csv.writer(open(f'data/value_points_corr.csv','w',newline=''))
    items = ['blood','aqp4igg','vitd','t_b','ig3']
    title = ['id','time','gender','age','state']
    for i in items:
        for j in range(len(default_data[i])):
            title.append(f'{i}-{j+1}')
    title.append('interval')
    title.append('event')
    # title.append('miss_info')
    w.writerow(title)

    positive, negative = 0, 0 #数据中尾时间点为发作的时间点数量
    for p in patients:
        p.make_value_points(item_threshold,time_threshold,corr,time)
        positive += sum([1 for i in p.value_points if i[-1]==1])
        negative += sum([1 for i in p.value_points if i[-1]==0])
        for v in p.value_points:
            w.writerow(v)
    print(f'Total number of value point example:{positive + negative}, positive:{positive}, negative:{negative}')
#构造点序列数据，存储在point_sequences.csv中
def make_point_seq():
    positive, negative = 0, 0
    f = open(f'data/point_sequences.csv', 'w', newline='')
    w = csv.writer(f)
    df = pd.read_csv(f'data/value_points.csv') #存储了有价值点信息

    id = -1
    temp = []
    series = []
    for (index, data) in df.iterrows():         #先把所有同一个id的有价值点组织到一起
        if id == -1 or data[0]== id:  
            if id == -1:
                id = data[0]
            temp.append(index)
        else:
            vp = {'id':id,'index':temp}
            id = data[0]
            temp = [index]
            series.append(vp)
    series.append({'id':id, 'index':temp})


    for vp in series:
        for i in range(len(vp['index'])-1):
            row = [vp['index'][i]]
            for j in range(i+1,len(vp['index'])):
                row.append(vp['index'][j])
                w.writerow([vp['id']] + row + [df.iloc[vp['index'][j]][-2],df.iloc[vp['index'][j]][-1] == 1])
                if df.iloc[vp['index'][j]][-1] == 1:
                    positive += 1
                else:
                    negative += 1
    print(f'Total number of point sequence examples:{positive + negative}, positive:{positive}, negative:{negative}')


def build_data_seq(id, start, end, duration, event, p):
    data = {'id': id}
    for k,v in p.records.items():
        temp = []
        for r in v:
            if dis(start, r[0])>=-30 and dis(r[0], end)>=-30:
                temp.append(r[1:])
        data[k] = temp
    data['duration'] = duration
    data['event'] = event
    return data

def make_data_seq(patients):
    f = open('data/data_sequence.txt', 'w')
    points = pd.read_csv(f'data/value_points.csv')
    sequences = open('data/point_sequences.csv').readlines()
    for line in sequences:
        temp = line.strip().split(',')
        id = int(points.iloc[int(temp[1])][0])
        start = points.iloc[int(temp[1])][1]
        end = points.iloc[int(temp[-3])][1]
        duration = int(temp[-2])
        event = int(eval(temp[-1]))
        p = patients[id]
        data = build_data_seq(id,start,end,duration,event,p)
        f.write(str(data) + '\n')
    f.close()



    
if __name__ == '__main__':
    patients_f = open('data/washed_sheet/patients.csv')
    patients = []
    for line in patients_f:
        patients.append(patient(line.strip()))
    print(f'Total number of item record:{sum([p.record_num() for p in patients])}')
    make_point_data(patients)
    make_point_seq()
    # make_data_seq(patients)

