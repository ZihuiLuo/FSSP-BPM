import numpy as np
import pandas as pd

def seem_equal(a, b):
    return sum(abs(a - b)) < 1e-5
def path2csv(path):
    columns = ['TOT_WGT','DEL_TO_DATE','PROCESS_TIME']
    df = pd.DataFrame(path, columns=columns)

    # 反正规化
    without_norm = pd.read_csv("../data/A_test2_without_norm.csv")
    norm_inverse = lambda x, low, high: (x * (high - low) + low)
    low = {name: without_norm[name].min() for name in columns[1:]}
    high = {name: without_norm[name].max() for name in columns[1:]}
    print("low:",low)
    print("high:",high)
    new_df = {name: norm_inverse(df[name], low[name], high[name]) for name in columns[1:]}
    new_df = pd.DataFrame(new_df)
    new_df['TOT_WGT'] = df['TOT_WGT']
    #print(new_df)
    new_df.loc[new_df['TOT_WGT']==0,:] = 0
    print(new_df)
    new_df.to_csv("flow_shop_result.csv")


list=[[[1.6959999799728394, 0.8041136264801025, 0.0], [7.271999835968018, 0.48971596360206604, 0.1428571492433548], [9.869999885559082, 0.0, 0.1428571492433548], [64.87200164794922, 0.901077389717102, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [2.7920000553131104, 0.19588638842105865, 0.1428571492433548], [2.4019999504089355, 1.0, 0.8571428656578064], [17.711999893188477, 0.4113613963127136, 0.5714285969734192], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[6.9720001220703125, 0.7051910161972046, 0.8571428656578064], [17.145000457763672, 0.5102840065956116, 0.2857142984867096], [1.3819999694824219, 0.4113613963127136, 0.7142857313156128], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]

array = np.array(list)
depot_num,node_num,input_dim =array.shape
print(array.shape)
total=[]
for i in range(depot_num):
    temp = []
    for j in range(node_num):
        if not(seem_equal(array[i][j], input_dim * [0.])):
            temp.append((array[i][j]).tolist())
        if j==node_num-1:
            temp.sort(key=(lambda x: x[1]))
            temp.append([0,0,0])

    total.append(temp)
print("total=",total)
final=[]
for i in range(len(total)):
    for j in range(len(total[i])):
        final.append(total[i][j])
print(final)

path2csv(final)
