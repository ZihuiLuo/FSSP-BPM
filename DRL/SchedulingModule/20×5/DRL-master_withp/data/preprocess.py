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
    stage_num = 5
    order = ['demand']
    p = ['p' + str(i + 1) for i in range(stage_num)]
    order.extend(p)

    total_size = 50000
    path = "fake_data.csv"

    df = pd.DataFrame(columns=order)
    df['demand'] = np.ones(total_size)*10
    for i in range(stage_num):
        df['p'+str(i+1)] = np.random.randint(1,99,size=(total_size))

    df.to_csv(path)

def preprocess2():
    # 从产生的fake_data中生成训练数据（正规化）
    stage_num = 5
    fake_data = pd.read_csv('fake_data.csv')
    print(fake_data.head())
    order = ['demand']
    p = ['p'+str(i+1) for i in range(stage_num)]
    order.extend(p)
    print(order)

    #test_without_norm = fake_data.sample(n=50)
    train_without_norm = fake_data

    norm = lambda x: (x - x.min()) / (x.max() - x.min())

    train_with_norm = train_without_norm[p].apply(norm)
    train_with_norm['demand'] = train_without_norm['demand']
    train_with_norm = train_with_norm[order]

    csv2txt('train.txt', train_with_norm)
    #csv2txt('test.txt', test_with_norm)


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

def preprocess3():
    # 正规化测试数据
    stage_num = 5
    order = ['demand']
    p = ['p' + str(i + 1) for i in range(stage_num)]
    order.extend(p)

    test_without_norm = pd.read_csv('test_without_norm.csv')

    norm = lambda x: (x - x.min()) / (x.max() - x.min())

    test_with_norm = test_without_norm[p].apply(norm)
    test_with_norm['demand'] = test_without_norm['demand']

    test_with_norm = test_with_norm[order]

    csv2txt('test.txt', test_with_norm)
#preprocess3()

#create_fake_data()
#preprocess2()
preprocess3()
#preprocess3()