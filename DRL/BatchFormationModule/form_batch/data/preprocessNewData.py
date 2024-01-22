import pandas as pd
import numpy as np
import os
import tensorflow as tf


# 根据一炉多标准表格 给定各个钢种优先级
from numpy import random


def fun1(x):
    if (x=='B16070GGALR')|(x=='B06108FFJCZ')|(x=='B14130FFZLR')|(x=='B16140GFNBU')|(x=='B07167HGCRB')|(x=='E18289BAHAN')|(x=='B17153HHCEB')|(x=='B16146GFFVR')|(x=='C16169B5FBY'):
        return 1
    if (x=='B14100GGFAH')|(x=='B06108FFJCD')|(x=='E18355CAFAJ')|(x=='B16140GFNBR')|(x=='B07159HGCRB')|(x=='E10145HHCVB')|(x=='B16126HHCEB')|(x=='B16144GFFVR')|(x=='B07110FFJCF'):
        return 2
    if (x=='B14085GGFAH')|(x=='A16090HHCN0')|(x=='B14130FFJLR')|(x=='B18355CAFAJ')|(x=='B16140GFALU')|(x=='B07157HGCRB')|(x=='E18279BAHAN')|(x=='B16116HHCEB'):
        return 3
    if (x == 'A15070JJNON') | (x == 'B16141HHCEB') | (x == 'B16140GFALR') :
        return 4
    if (x == 'A16050JJNON') | (x == 'A17140MLNNA') | (x == 'B18345CAFAJ') :
        return 5
    if (x == 'A16050JJNNG') | (x == 'B15130HHCEB') | (x == 'B17131HHCNB') :
        return 6
    if (x == 'B14120FFJLR') | (x == 'B16135ZHCNb') :
        return 7
    if (x == 'A17100MLNNA') | (x == 'B15130JJCNb') :
        return 8
    if x == 'B14120FFJLV':
        return 9


# 根据一炉多标准表格，给定各个钢种具体钢种组
def fun2(x):
    if (x=='B16070GGALR')|(x=='B14100GGFAH')|(x=='B14085GGFAH')|(x=='A15070JJNON')|(x=='A16050JJNON')|(x=='A16050JJNNG'):
        return 'A'
    if (x=='B06108FFJCZ')|(x=='B06108FFJCD')|(x=='A16090HHCN0'):
        return 'B'
    if (x=='B14130FFZLR')|(x=='B14130FFJLV')|(x=='B14130FFJLR')|(x=='B16141HHCEB')|(x=='A17140MLNNA')|(x=='B15130HHCEB')|(x=='B14120FFJLR')|(x=='A17100MLNNA')|(x=='B14120FFJLV'):
        return 'C'
    if (x=='E18355CAFAJ')|(x=='B18355CAFAJ')|(x=='B18345CAFAJ')|(x=='B17131HHCNB')|(x=='B16135ZHCNb')|(x=='B15130JJCNb'):
        return 'D'
    if (x=='B16140GFNBU')|(x=='B16140GFNBR')|(x=='B16140GFALU')|(x=='B16140GFALR'):
        return 'E'
    if (x=='B07167HGCRB')|(x=='B07159HGCRB')|(x=='B07157HGCRB'):
        return 'F'
    if (x=='E18289BAHAN')|(x=='E10145HHCVB')|(x=='E18279BAHAN'):
        return 'G'
    if (x=='B17153HHCEB')|(x=='B16126HHCEB')|(x=='B16116HHCEB'):
        return 'H'
    if (x=='B16146GFFVR')|(x=='B16144GFFVR'):
        return 'I'
    if (x=='C16169B5FBY')|(x=='B07110FFJCF'):
        return 'Z'

# 预处理定序型数据
def fun3(data_path):
    all_data = pd.read_excel(data_path)
    #print(all_data.head(10))
    #print(all_data.info())

    # 厚度
    all_data['thickness'] = all_data['thickness'].map({150:1,220:2,260:3,320:4})
    # 工艺
    all_data['art'] = all_data['art'].map({'Y': 1, 'N':0})
    return all_data

def fun4(all_data):
    #all_data = pd.read_excel(data_path)
    all_data['priority'] = all_data['class'].apply(lambda x: fun1(x))
    all_data['class_group'] = all_data['class'].apply(lambda x: fun2(x))

    print(all_data.info())
    return all_data

# 将钢种组 转成one-hot类型
def fun5(data_path):
    all_data = pd.read_excel(data_path)
    c = pd.get_dummies(all_data['class_group'])
    all_data = pd.concat([all_data,c],axis = 1)
    print(all_data.info())
    new_data_path = '/Users/jiangchengling/Desktop/all_data/form_batch_data/test.xlsx'
    all_data.to_excel(new_data_path)

def fun6(data_path):
    all_data = pd.read_excel(data_path)
    print(all_data['date'].describe())

# 该方法将要求的 属性进行正规化
def fun7():
    data_path = '/Users/jiangchengling/Desktop/all_data/form_batch_data/train_data_date.xlsx'
    new_data_path = '/Users/jiangchengling/Desktop/all_data/form_batch_data/normed_train_data.xlsx'
    all_data = pd.read_excel(data_path)
    norm = lambda x: (x - x.min()) / (x.max() - x.min())
    all_data[['thickness', 'width', 'date', 'priority', 'length']] = all_data[['thickness', 'width', 'date', 'priority', 'length']].apply(norm, axis=0)
    all_data.to_excel(new_data_path)

# 根据测试数据的id，从正规化后的训练数据里面抽取正规化后的测试数据
def fun8():
    h1 = pd.read_excel('/Users/jiangchengling/Desktop/all_data/form_batch_data/origin_test_data.xlsx')
    h2 = pd.read_excel('/Users/jiangchengling/Desktop/all_data/form_batch_data/normed_train_data.xlsx')
    test = pd.merge(left=h1,right=h2,how='left',on=['id'])
    test.to_excel('/Users/jiangchengling/Desktop/all_data/form_batch_data/normed_test_data.xlsx')

# 将日期数据转换成与2019-12-31的天数差值
def fun9():
    data_path = '/Users/jiangchengling/Desktop/all_data/form_batch_data/train_data.xlsx'
    all_data = pd.read_excel(data_path)
    print(all_data.dtypes)
    # print(all_data['date'].min())
    # print(all_data['date'].max())
    all_data['date']=all_data['date'].astype('object')
    all_data['date'] = pd.to_datetime(all_data['date'],format = '%Y%m%d')
    data_new = pd.to_datetime('2019-12-31')
    all_data['date'] = ((all_data['date']-data_new)/np.timedelta64(1, 'D')).astype('int')
    print(all_data.dtypes)
    all_data.to_excel('/Users/jiangchengling/Desktop/all_data/form_batch_data/train_data_date.xlsx')

def fun10():
    data_path = '/Users/jiangchengling/Desktop/all_data/form_batch_data/normed_train_data.xlsx'
    all_data = pd.read_excel(data_path)
    all_data['id_2'] = all_data['id'].astype(str)
    print(all_data.info())
    all_data.to_excel('/Users/jiangchengling/Desktop/all_data/form_batch_data/normed_train_data2.xlsx')


#将excel数据 转成txt
def csv2txt():
    data_path = '/Users/jiangchengling/Desktop/all_data/form_batch_data/normed_train_data2.xlsx'
    data = pd.read_excel(data_path)
    print(data.info())
    data['id'] = data['id']-2010000000
    data['id'] = data['id'].astype(str)
    print(data.info())
    filename = 'train.txt'
    file = open(filename, 'w', encoding='utf-8')
    for line in data.values:
        for word in line:
            file.write(f'{word}\t')
        file.write('\n')


class DataGenerator():
    def __init__(self):
        self.train_data = np.loadtxt('new_train.txt')
        self.count = 0
        self.batch_size = 1
        self.n_cust = 30
        self.input_dim = 16

    def shuffle_data(self):
        # shuffle_data在每个epoch开始都执行
        row_rand_array = np.arange(self.train_data.shape[0])
        np.random.shuffle(row_rand_array)
        self.train_data = self.train_data[row_rand_array]
        self.count = 0

    def get_train_batch(self):

        train_batch = self.train_data[self.count:(self.count+self.batch_size* self.n_cust)]
        self.count += self.count+self.batch_size* self.n_cust

        cust_data = np.array(train_batch.values.tolist()).reshape(self.batch_size, self.n_cust, self.input_dim)
        depot = np.zeros((self.batch_size, 1, self.input_dim))
        batch_data = np.concatenate([cust_data, depot], 1)

        return batch_data

csv2txt()



#data_path = '/Users/jiangchengling/Desktop/all_data/form_batch_data/train2.xlsx'
# all_data = fun3(data_path) # thickness、 art
#all_data = fun5(data_path)
# all_data = all_data.dropna()
# all_data.to_excel('/Users/jiangchengling/Desktop/all_data/form_batch_data/train3.xlsx')

#data_path = '/Users/jiangchengling/Desktop/all_data/form_batch_data/origin_test_data.xlsx'
#fun5(data_path)
#fun6(data_path)


# with tf.Graph().as_default():
#     sess = tf.Session()
#     a = tf.constant([1,2,3,4,5])
#     b= tf.expand_dims(a,1)
#     c = tf.tile(tf.expand_dims(a,1),[1,3])
#     print("b:\n",sess.run(b))
#     print("c:\n", sess.run(c))

