import pandas as pd
import os
import numpy as np

def csv2txt(filename,data):
    file = open(filename, 'w', encoding='utf-8')
    for line in data.values:
        for word in line:
            file.write(f'{word}\t')
        file.write('\n')

def create_fake_data():
    import pandas as pd
    import numpy as np

    total_size = 100000
    path = "fake_data2.csv"

    df = pd.DataFrame(columns=[ 'ORD_WID', 'ORD_THK','priority','PROCESS_TIME','CAN_START_TIME','TOT_WGT'])

    df['ORD_WID'] = np.random.randint(1500, 2500, size=total_size)
    df['ORD_THK'] = np.random.randint(5, 60, size=total_size)
    df['priority'] = np.random.randint(1, 9, size=total_size)
    df['PROCESS_TIME'] = np.random.randint(2, 15, size=total_size)
    df['CAN_START_TIME'] = np.random.randint(0, 20, size=total_size)
    df['TOT_WGT'] = np.random.uniform(low=1.0, high=50.0, size=total_size)

    df.to_csv(path)

def preprocess2():
    fake_data = pd.read_csv('fake_data2.csv')
    print(fake_data.head())
    order = [ 'ORD_WID', 'ORD_THK','priority','PROCESS_TIME','CAN_START_TIME','TOT_WGT']
    train_without_norm = fake_data.sample(n=50000)
    train_without_norm = train_without_norm[order]
    csv2txt('train.txt', train_without_norm)

    test_without_norm = fake_data.sample(n=50)
    test_without_norm.to_csv("test_without_norm.csv")
    norm = lambda x: (x - x.min()) / (x.max() - x.min())

    # train = train_without_norm[[ 'ORD_WID', 'ORD_THK','priority','PROCESS_TIME','CAN_START_TIME']].apply(norm, axis=0)
    # train['TOT_WGT'] = train_without_norm['TOT_WGT']
    test = test_without_norm[['ORD_WID', 'ORD_THK','priority','PROCESS_TIME','CAN_START_TIME']].apply(norm, axis=0)
    test['TOT_WGT'] = test_without_norm['TOT_WGT']

    order = ['ORD_WID', 'ORD_THK','priority','PROCESS_TIME','CAN_START_TIME','TOT_WGT']
    test = test[order]
    csv2txt('test.txt', test)

def preprocess():
    # 从钢种得到钢种组,并作为one_hot多列,删除无钢组的行
    all_data = pd.read_csv('all_data.csv', usecols=['TOT_WGT', 'STLGRD', 'DEL_TO_DATE', 'ORD_WID', 'ORD_THK'])
    print(f'length of all data:{len(all_data)}')
    type_group = pd.read_csv('type_group.csv', usecols=['钢种组', '优先顺序', '钢种'])
    type_group.rename(columns={'钢种组': 'group', '优先顺序': 'priority', '钢种': 'STLGRD'}, inplace=True)
    # 添加钢组
    df = pd.merge(all_data, type_group, on='STLGRD', how='left')
    # 删除无钢组的行
    df = df[df['group'].notnull()]

    df = df[df['group'] == 'A']
    df = df.drop(columns=['group', 'STLGRD'])

    # 去重
    df = df.drop_duplicates(list(df.columns)[1:])
    # 去掉需求量大于50的行
    train_without_norm = df[df['TOT_WGT'] <= 50]
    print(f'after deduplicate, length of data:{len(df)}')

    # train_without_norm.to_csv("A_train_without_norm.csv")
    test_without_norm = train_without_norm.sample(n=50)
    test_without_norm.to_csv("A_test_without_norm.csv")
    # 正规化
    norm = lambda x: (x - x.min()) / (x.max() - x.min())
    train = train_without_norm[['DEL_TO_DATE','ORD_WID','ORD_THK','priority']].apply(norm,axis=0)
    train['TOT_WGT'] = train_without_norm['TOT_WGT']
    test = test_without_norm[['DEL_TO_DATE','ORD_WID','ORD_THK','priority']].apply(norm, axis=0)
    test['TOT_WGT'] = test_without_norm['TOT_WGT']
    # pd.DataFrame(train).to_csv("A_train.csv")
    # pd.DataFrame(test).to_csv("A_test.csv")
    csv2txt('train.txt', train)
    csv2txt('test.txt', test)

# 从结果反推出test.txt
def result2test():
    result_without_norm = pd.read_csv("result(1)(1).csv",usecols=['DEL_TO_DATE', 'ORD_WID', 'ORD_THK', 'priority','load'])

    decompose_row = result_without_norm[pd.Series(map(lambda str:str.startswith("Decompose"),result_without_norm['load']))]

    seem_equal = lambda a, b:sum(abs(a - b)) < 1e-5
    for i in range(len(result_without_norm)):
        for j in range(len(decompose_row)):
            if(seem_equal(result_without_norm.iloc[i,:-1],decompose_row.iloc[j,:-1])):
                result_without_norm.iloc[i,-1]="Group"
    group_row = pd.Series(map(lambda str: not str.startswith("Group"), result_without_norm['load']))
    result_without_norm = result_without_norm[group_row]
    result_without_norm['load']=result_without_norm['load'].apply(lambda str:float(str.split()[0]))
    decompose_row['load']=decompose_row['load'].apply(lambda str:float(str.split()[1][:-1]))
    result_without_norm = pd.concat([result_without_norm,decompose_row])
    result_without_norm.to_csv("tmp.csv")

    norm = lambda x: (x - x.min()) / (x.max() - x.min())
    result = result_without_norm[['DEL_TO_DATE', 'ORD_WID', 'ORD_THK', 'priority']].apply(norm, axis=0)
    result['TOT_WGT'] = result_without_norm['load']
    csv2txt('test.txt', result)

# preprocess2()
