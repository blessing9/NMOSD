import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import csv
import torch
from utils import *
from dataset import *
from model import Model
from torch import nn
import random





def train(args, model, train_set, test_set, i):
    lr = args.learning_rate
    epoches = args.num_train_epoches

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = test_set.__len__(),shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0.0, epoches)

    loss_function = neg_loss if args.task.__contains__('hazard') else nn.MSELoss()
    metric_function = c_index if args.task.__contains__('hazard') else auc

    #记录训练过程中训练集和测试集loss及metric的变化
    all_train_loss, all_train_metric, all_test_loss, all_test_metric = [],[],[],[]
    for epoch in range(epoches):
        epoch_pred, epoch_label, epoch_time, epoch_event, epoch_loss, epoch_count = [], [], [],[], 0, 0
        for x, y in train_loader:
            if args.task.__contains__('hazard') and y[1].sum() == 0:    #对于风险预测任务，如果全部是缺失数据，无法计算loss，跳过
                continue
            optimizer.zero_grad()
            o = model(x)
            loss = loss_function(o, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            epoch_count += o.shape[0]
            epoch_pred += o.tolist()
            if args.task.__contains__('hazard'):
                epoch_time += y[0].tolist()
                epoch_event += y[1].tolist()
            else:
                epoch_label += y.tolist()
        epoch_label = epoch_label if not args.task.__contains__('hazard') else [epoch_time, epoch_event]
        epoch_metric = metric_function(epoch_pred, epoch_label)
        epoch_loss /= epoch_count
        all_train_metric.append(epoch_metric)
        all_train_loss.append(epoch_loss)

        #测试
        with torch.no_grad():
            for x, y in test_loader:
                o = model(x)
                test_metric = metric_function(o, y)
                all_test_metric.append(test_metric)
                all_test_loss.append(loss_function(o, y).item())
        if epoch % 10 == 9:
            print(f'epoch[{epoch + 1}/{epoches}] loss={epoch_loss}, train c-index={epoch_metric}, test c-index={test_metric}', end='\r')
    with torch.no_grad():
        for x, y in test_loader:
            o = model(x)
            test_metric = metric_function(o, y)
    print(f'[fold {i+1}] train_c = {epoch_metric}, test_c = {test_metric}')

    x = list(range(1, epoches +1))
    plt.scatter(x, all_test_metric)
    plt.scatter(x, all_train_metric)
    plt.savefig(f'pictures/{args.task}_training_process_{i}.jpg')
    plt.close()
    return epoch_metric, test_metric
    







if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #task
    parser.add_argument('--task', default='point_seq_hazard', type=str, choices=['point_hazard','point_binary', 'point_seq_hazard', 'point_seq_binary', 'data_seq_hazard', 'data_seq_binary'])

    #dataset file
    parser.add_argument('--value_point_file', default='data/value_points.csv', type=str)
    parser.add_argument('--point_seq_file', default='data/point_sequences.csv', type=str)
    parser.add_argument('--data_seq_file', default='data/data_sequence.txt', type=str)

    #model setting
    parser.add_argument('--hidden_size_point_seq', default=256, type=int) #hidden_size of LSTM for pint_seq task
    parser.add_argument('--fc_point_seq', default= [256,512,256,1])#fully-connected network setting for point_seq task
    parser.add_argument('--fc_point', default=[128,256,128, 1]) #fully-connected network setting for point task
    parser.add_argument('--fc_data_seq',default=[256,512,256,128,1]) #fully-connected network setting for data_seq task
    parser.add_argument('--hidden_size_data_seq', default=[256, 256, 256, 256, 256]) #hidden_size of LSTM for data_seq task

    #train setting
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--fold', default=5, type=int)
    parser.add_argument('--num_train_epoches', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)

    #random seed
    parser.add_argument('--seed', default=1111, type=int)

    args = parser.parse_args()

    #固定随机种子
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #dataset
    data_set = Dataset(args)
    args.input_size = data_set.input_size

    #experiment result
    all_train_metrics, all_test_metrics = [], []
    result = csv.writer(open(f'result/result_{args.task}.csv','w', newline=''))

    #cross-validation
    for i in range(args.fold):
        #model
        model = Model(args).to(args.device)
        train_set, test_set = get_train_test(data_set, i)
        train_metrics, test_metrics = train(args, model, train_set, test_set, i)
        all_train_metrics.append(round(train_metrics, 3))
        all_test_metrics.append(round(test_metrics, 3))
    result.writerow([''] + [str(i) for i in range(1, 6)] + ['mean'])
    result.writerow(['train'] + all_train_metrics + [round(np.mean(all_train_metrics),3 )])
    result.writerow(['test'] + all_test_metrics + [round(np.mean(all_test_metrics), 3)])



