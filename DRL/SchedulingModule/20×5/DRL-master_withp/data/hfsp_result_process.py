import numpy as np
import pandas as pd
import tensorflow as tf


def seem_equal(a, b):
    return sum(abs(a - b)) < 1e-5


# 本方法将归一化input test data反正规化，还原并返回原始数据
def norm_inverse(path,stage):
    stage_num = stage
    columns = ['demand']
    p = ['p' + str(i + 1) for i in range(stage_num)]
    columns.extend(p)
    #columns = ['demand','p1','p2','p3','p4','p5']
    df = pd.DataFrame(path, columns=columns)
    # 反正规化
    without_norm = pd.read_csv("test_without_norm.csv")
    norm_inverse = lambda x, low, high: (x * (high - low) + low)
    low = {name: without_norm[name].min() for name in columns[1:]}
    high = {name: without_norm[name].max() for name in columns[1:]}
    new_df = {name: norm_inverse(df[name], low[name], high[name]) for name in columns[1:]}
    new_df = pd.DataFrame(new_df)
    new_df['demand'] = df['demand']
    #order = ['demand','p1','p2','p3','p4','p5']
    new_df = new_df[columns]
    return new_df


def update_start_end_time1(idx, stage):
    global end_time, start_time, depot_num
    """
    end_time: tensor of shape (stage_num, batch_size, cust_num)
    process_time: tensor of shape (batch_size, cust_num, stage_num)
    idx: cust_num selected, tensor of shape (depot_num, node_num, batch_size)
    stage: int
    """

    # # print(self.input_data)
    # # batch_size, node,_ = self.input_data.shape.as_list()
    # print("batch_size=",batch_size)
    def modify1(tensor, index, value):
        shape = tf.shape(tensor)
        return tensor - tensor[index] * tf.one_hot(index, shape[0]) + value * tf.one_hot(index, shape[0])

        # tensor，index都是二维，value标量

    def modify2(tensor, index, value):
        return tf.concat([tensor[:index[0]],
                          tf.expand_dims(
                              modify1(tensor[index[0]], index[1], value), axis=0),
                          tensor[index[0] + 1:]],
                         axis=0)

        # tensor，index都是三维，value标量

    def modify3(tensor, index, value):
        return tf.concat([tensor[:index[0]],
                          tf.expand_dims(
                              modify2(tensor[index[0]], index[1:], value), axis=0),
                          tensor[index[0] + 1:]],
                         axis=0)

    batch = 0
    n = batch_size

    def cond(batch, n, end_time, start_time):
        return batch < n

    def body(batch, n, end_time, start_time):

        node_list = idx[:, batch]


        node_list = tf.map_fn(
            lambda node: tf.cast(node, dtype=tf.int32),
            node_list, dtype=tf.int32)


        process_time_node = tf.map_fn(lambda node: process_time[batch, node, stage],
                                      node_list, dtype=tf.float32)
        # 选择的订单对应的结束时间，下标为node
        end_time_node = tf.zeros(node_num - 1)

        end_time_node = modify1(end_time_node, 0,
                                tf.reduce_max(
                                    [(end_time[
                                          stage - 1, batch, node_list[0]] if stage != 0 else 0.)
                                        , 0.], axis=0)
                                + process_time_node[0])
        # 第i个结点的结束时间为（max（自身上一个阶段的结束时间，上一个结点的结束时间）+ 其process_time）
        for i in range(1, node_num - 1):
            end_time_node = modify1(end_time_node, i,
                                    tf.reduce_max(
                                        [(end_time[
                                              stage - 1, batch, node_list[
                                                  i]] if stage != 0 else 0.),
                                         end_time_node[i - 1]],
                                        axis=0) +
                                    process_time_node[i])

        for i in range(node_num - 1):
            end_time = modify3(end_time, (stage, batch, node_list[i]), end_time_node[i])
            start_time = modify3(start_time, (stage, batch, node_list[i]),
                                 end_time_node[i] - process_time_node[i])

        batch = batch + 1
        return batch, n, end_time, start_time

    batch, n, end_time, start_time = tf.while_loop(cond, body, [batch, n, end_time, start_time])


# 测试时正规化的输入数据
list =[[10.0, 0.23469388, 0.5102041, 0.071428575, 0.59183675, 0.67346936], [10.0, 0.0, 0.5714286, 0.5408163, 0.85714287, 1.0], [10.0, 0.5714286, 0.82653064, 0.14285715, 0.7346939, 0.5714286], [10.0, 0.86734694, 0.46938777, 0.46938777, 0.877551, 0.46938777], [10.0, 0.3469388, 0.82653064, 0.7346939, 0.877551, 0.93877554], [10.0, 0.08163265, 0.13265306, 0.3265306, 0.8979592, 0.3469388], [10.0, 0.5510204, 0.79591835, 0.7346939, 0.47959185, 0.31632653], [10.0, 0.071428575, 0.5714286, 0.3469388, 0.5714286, 0.7755102], [10.0, 0.93877554, 0.8877551, 0.4489796, 0.59183675, 0.2755102], [10.0, 0.10204082, 0.79591835, 0.75510204, 0.7346939, 0.75510204], [10.0, 0.3877551, 0.21428572, 0.8367347, 0.6632653, 0.9591837], [10.0, 0.091836736, 0.7244898, 0.1122449, 0.9591837, 0.14285715], [10.0, 0.8469388, 0.5816327, 0.59183675, 0.6122449, 0.1734694], [10.0, 0.8061224, 0.78571427, 0.9591837, 0.12244898, 0.97959185], [10.0, 0.59183675, 0.1734694, 0.96938777, 0.79591835, 0.36734694], [10.0, 0.36734694, 0.47959185, 0.75510204, 0.8877551, 0.53061223], [10.0, 0.8061224, 0.7244898, 0.622449, 0.18367347, 0.97959185], [10.0, 0.6938776, 0.37755102, 0.4387755, 0.1734694, 0.7346939], [10.0, 0.12244898, 0.71428573, 0.42857143, 0.20408164, 0.6020408], [10.0, 0.18367347, 0.82653064, 0.29591838, 0.3265306, 0.3877551], [10.0, 0.24489796, 0.6122449, 0.46938777, 0.10204082, 0.47959185], [10.0, 0.9285714, 0.7346939, 0.35714287, 0.071428575, 0.08163265], [10.0, 0.4489796, 0.5, 0.6530612, 0.071428575, 0.1632653], [10.0, 0.13265306, 0.56122446, 0.85714287, 0.8367347, 0.8367347], [10.0, 0.7346939, 0.9591837, 0.0, 0.12244898, 0.091836736], [10.0, 0.85714287, 0.15306123, 0.71428573, 0.8979592, 0.010204081], [10.0, 0.75510204, 0.05102041, 0.12244898, 0.63265306, 0.6632653], [10.0, 0.96938777, 0.06122449, 0.5408163, 0.64285713, 0.5], [10.0, 0.3265306, 0.86734694, 0.010204081, 0.53061223, 0.35714287], [10.0, 0.1632653, 0.20408164, 0.82653064, 0.2755102, 0.4489796], [10.0, 0.7653061, 0.26530612, 0.15306123, 0.1632653, 0.86734694], [10.0, 0.96938777, 0.4387755, 0.14285715, 0.010204081, 0.071428575], [10.0, 0.030612245, 0.64285713, 0.13265306, 0.7755102, 0.46938777], [10.0, 0.15306123, 0.31632653, 0.48979592, 0.6122449, 0.52040815], [10.0, 0.08163265, 0.45918366, 0.68367344, 0.7346939, 0.9897959], [10.0, 0.6632653, 0.36734694, 0.2755102, 0.6938776, 0.21428572], [10.0, 0.45918366, 0.3877551, 0.877551, 0.2244898, 0.5510204], [10.0, 0.64285713, 0.96938777, 0.85714287, 0.56122446, 0.622449], [10.0, 0.020408163, 0.4387755, 0.90816325, 0.78571427, 0.2244898], [10.0, 0.93877554, 0.70408165, 0.1632653, 0.6530612, 0.82653064], [10.0, 0.74489796, 0.40816328, 0.5408163, 0.85714287, 0.31632653], [10.0, 0.2857143, 0.56122446, 0.59183675, 1.0, 0.19387755], [10.0, 0.68367344, 0.4387755, 0.24489796, 0.52040815, 0.37755102], [10.0, 0.37755102, 0.26530612, 0.2244898, 0.78571427, 0.2244898], [10.0, 0.86734694, 0.40816328, 0.6020408, 0.071428575, 0.39795917], [10.0, 0.90816325, 0.6632653, 0.6020408, 0.5816327, 0.5510204], [10.0, 0.5, 0.877551, 0.68367344, 0.5102041, 0.79591835], [10.0, 0.10204082, 0.41836736, 0.70408165, 0.42857143, 0.6632653], [10.0, 0.64285713, 0.06122449, 0.8061224, 0.52040815, 0.97959185], [10.0, 0.030612245, 0.5408163, 0.030612245, 0.40816328, 0.2857143], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]


# 各配送中心各阶段作业加工顺序
idx=[[1, 34, 5, 7, 23, 10, 32, 47, 33, 0, 9, 4, 48, 38, 29, 15, 18, 49, 46, 11, 26, 41, 43, 2, 30, 20, 19, 16, 17, 14, 39, 36, 13, 28, 27, 3, 40, 37, 35, 42, 6, 45, 22, 25, 44, 12, 8, 24, 21, 31], [1, 34, 5, 7, 23, 10, 32, 47, 33, 0, 9, 4, 48, 38, 29, 15, 18, 49, 46, 11, 26, 41, 43, 2, 30, 20, 19, 16, 17, 14, 39, 36, 13, 28, 27, 3, 40, 37, 35, 42, 6, 45, 22, 25, 44, 12, 8, 24, 21, 31], [1, 34, 5, 7, 23, 10, 32, 47, 33, 0, 9, 4, 48, 38, 29, 15, 18, 49, 46, 11, 26, 41, 43, 2, 30, 20, 19, 16, 17, 14, 39, 36, 13, 28, 27, 3, 40, 37, 35, 42, 6, 45, 22, 25, 44, 12, 8, 24, 21, 31], [1, 34, 5, 7, 23, 10, 32, 47, 33, 0, 9, 4, 48, 38, 29, 15, 18, 49, 46, 11, 26, 41, 43, 2, 30, 20, 19, 16, 17, 14, 39, 36, 13, 28, 27, 3, 40, 37, 35, 42, 6, 45, 22, 25, 44, 12, 8, 24, 21, 31], [1, 34, 5, 7, 23, 10, 32, 47, 33, 0, 9, 4, 48, 38, 29, 15, 18, 49, 46, 11, 26, 41, 43, 2, 30, 20, 19, 16, 17, 14, 39, 36, 13, 28, 27, 3, 40, 37, 35, 42, 6, 45, 22, 25, 44, 12, 8, 24, 21, 31]]

node_num = 51
cust_num = 50
stage_num = 5
batch_size = 1


array = np.array(list[:-1])
# origin_test_data为还原后的输入数据
origin_test_data = norm_inverse(array,stage_num)
print(origin_test_data)
# process_time为测试数据的各阶段加工时间（目前时2阶段）
process_time = []
p_time = origin_test_data.iloc[:, 1:].values.tolist()
process_time.append(p_time)
print("process_time:\n", p_time)

start_time = tf.zeros((stage_num, batch_size, cust_num))
end_time = tf.zeros((stage_num, batch_size, cust_num))
process_time = tf.convert_to_tensor(process_time)



sess = tf.Session()
for stage in range(stage_num):
    stage_id = tf.convert_to_tensor(idx[stage])
    stage_id = tf.expand_dims(stage_id, -1)
    update_start_end_time1(stage_id, stage)

#print("start_time:\n", sess.run(start_time))
#print("end_time:\n", sess.run(end_time))

print(sess.run(tf.reduce_max(end_time[-1],1)))
# 下面将start_time和end_time存入一维列表
st = np.round(np.squeeze(sess.run(start_time)))
et = np.round(np.squeeze(sess.run(end_time)))
p_array = np.round(np.array(p_time))
count = 0
# 这里解决batch_size=1的情况
machine_array = np.zeros((stage_num, cust_num))
for stage in range(stage_num):
    for index in range(node_num-1):
        if (idx[stage][index]) != cust_num:
            machine_array[stage][idx[stage][index]] = count
    count = count + 1

st_list = []
et_list = []
p_list = []
job_id_list = []
op_list = np.arange(cust_num)
n_bay_start = []

for i in range(stage_num):
    for j in range(cust_num):
        st_list.append(st[i][j])
        et_list.append(et[i][j])
        p_list.append(p_array[j, i])
        job_id_list.append(j)
        n_bay_start.append(machine_array[i][j])

df = pd.DataFrame({'Machine': n_bay_start,
                   'Start': st_list,
                   'Finish': et_list,
                   'Duration': p_list,
                   'job_id': job_id_list})
df.to_csv("gantt_data.csv")
