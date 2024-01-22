import numpy as np
import pandas as pd
import tensorflow as tf


def seem_equal(a, b):
    return sum(abs(a - b)) < 1e-5


# 本方法将归一化input test data反正规化，还原并返回原始数据
def norm_inverse(path):
    stage_num = 10
    columns = ['demand']
    p = ['p' + str(i + 1) for i in range(stage_num)]
    columns.extend(p)
    #columns = ['demand','p1','p2','p3','p4','p5']
    df = pd.DataFrame(path, columns=columns)
    # 反正规化
    without_norm = pd.read_csv("test_without_norm.csv")
    norm_inverse = lambda x: (x * (99 - 1) + 1)
    # low = {name: without_norm[name].min() for name in columns[1:]}
    # high = {name: without_norm[name].max() for name in columns[1:]}
    new_df = {name: norm_inverse(df[name]) for name in columns[1:]}
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
list = [[10.0, 0.48979592, 0.020618556, 0.6041667, 0.1122449, 0.75510204, 0.30612245, 0.7010309, 0.19387755, 0.8854167, 0.082474224], [10.0, 0.48979592, 0.53608245, 0.8958333, 0.46938777, 0.18367347, 0.6122449, 0.78350514, 0.78571427, 0.28125, 0.41237113], [10.0, 0.14285715, 0.5979381, 0.13541667, 0.45918366, 0.6020408, 0.46938777, 0.40206185, 0.37755102, 0.32291666, 0.21649484], [10.0, 0.1734694, 0.06185567, 0.25, 0.1632653, 0.877551, 0.31632653, 0.072164945, 0.25510204, 0.3125, 0.34020618], [10.0, 0.6530612, 0.6597938, 0.09375, 0.42857143, 0.26530612, 0.33673468, 0.46391752, 0.64285713, 0.1875, 0.8453608], [10.0, 0.5510204, 0.58762884, 0.14583333, 0.15306123, 0.47959185, 0.6938776, 0.9484536, 0.622449, 0.041666668, 0.556701], [10.0, 0.0, 0.2371134, 0.5520833, 0.9183673, 0.7244898, 0.31632653, 0.24742268, 0.10204082, 0.9479167, 0.020618556], [10.0, 0.79591835, 0.556701, 0.59375, 0.9489796, 0.8877551, 0.0, 0.6494845, 0.37755102, 0.072916664, 0.072164945], [10.0, 0.091836736, 0.25773194, 0.8229167, 0.7346939, 0.47959185, 0.0, 0.15463917, 0.68367344, 0.7291667, 0.7628866], [10.0, 0.36734694, 0.40206185, 0.53125, 0.8979592, 0.59183675, 0.8061224, 0.41237113, 0.36734694, 0.8541667, 0.29896906], [10.0, 0.7755102, 0.9072165, 0.9375, 0.1122449, 0.74489796, 0.18367347, 0.58762884, 0.74489796, 0.8229167, 0.57731956], [10.0, 0.8061224, 0.9587629, 0.21875, 0.5816327, 0.86734694, 0.5714286, 0.8041237, 0.08163265, 0.6145833, 0.6597938], [10.0, 0.79591835, 0.5154639, 0.5520833, 0.24489796, 0.48979592, 0.9897959, 0.05154639, 0.6530612, 1.0, 0.9072165], [10.0, 0.8469388, 0.7525773, 0.8020833, 0.23469388, 0.9489796, 0.36734694, 0.7525773, 0.15306123, 0.6666667, 0.6082474], [10.0, 0.93877554, 0.5463917, 0.114583336, 0.5510204, 0.14285715, 0.30612245, 0.62886596, 0.37755102, 0.6041667, 0.91752577], [10.0, 0.20408164, 0.87628865, 0.90625, 0.0, 0.9591837, 0.5102041, 0.96907216, 0.85714287, 0.072916664, 0.8556701], [10.0, 0.85714287, 0.21649484, 0.0, 0.67346936, 0.40816328, 0.6632653, 0.05154639, 0.5, 0.010416667, 0.7525773], [10.0, 0.64285713, 0.8453608, 0.15625, 0.020408163, 0.9489796, 0.37755102, 0.082474224, 0.622449, 0.6979167, 0.16494845], [10.0, 0.45918366, 0.185567, 0.9895833, 0.0, 0.14285715, 0.622449, 0.46391752, 0.3877551, 0.29166666, 0.010309278], [10.0, 0.3469388, 0.44329897, 0.40625, 0.71428573, 0.71428573, 0.7244898, 0.371134, 0.97959185, 0.47916666, 0.185567], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

# 各配送中心各阶段作业加工顺序
idx=[[3, 8, 2, 19, 6, 12, 15, 9, 14, 4, 5, 10, 0, 11, 16, 13, 1, 17, 18, 7], [3, 8, 2, 19, 6, 12, 15, 9, 14, 4, 5, 10, 0, 11, 16, 13, 1, 17, 18, 7], [3, 8, 2, 19, 6, 12, 15, 9, 14, 4, 5, 10, 0, 11, 16, 13, 1, 17, 18, 7], [3, 8, 2, 19, 6, 12, 15, 9, 14, 4, 5, 10, 0, 11, 16, 13, 1, 17, 18, 7], [3, 8, 2, 19, 6, 12, 15, 9, 14, 4, 5, 10, 0, 11, 16, 13, 1, 17, 18, 7], [3, 8, 2, 19, 6, 12, 15, 9, 14, 4, 5, 10, 0, 11, 16, 13, 1, 17, 18, 7], [3, 8, 2, 19, 6, 12, 15, 9, 14, 4, 5, 10, 0, 11, 16, 13, 1, 17, 18, 7], [3, 8, 2, 19, 6, 12, 15, 9, 14, 4, 5, 10, 0, 11, 16, 13, 1, 17, 18, 7], [3, 8, 2, 19, 6, 12, 15, 9, 14, 4, 5, 10, 0, 11, 16, 13, 1, 17, 18, 7], [3, 8, 2, 19, 6, 12, 15, 9, 14, 4, 5, 10, 0, 11, 16, 13, 1, 17, 18, 7]]


node_num = 21
cust_num = 20
stage_num = 10
batch_size = 1


array = np.array(list[:-1])
# origin_test_data为还原后的输入数据
origin_test_data = norm_inverse(array)
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

# print("start_time:\n", sess.run(start_time))
# print("end_time:\n", sess.run(end_time))

print(sess.run(tf.reduce_max(end_time[-1],1)))
# 下面将start_time和end_time存入一维列表
# st = np.round(np.squeeze(sess.run(start_time)))
# et = np.round(np.squeeze(sess.run(end_time)))
# p_array = np.round(np.array(p_time))
# count = 0
# # 这里解决batch_size=1的情况
# machine_array = np.zeros((stage_num, cust_num))
# for stage in range(stage_num):
#     for index in range(node_num-1):
#         if (idx[stage][index]) != cust_num:
#             machine_array[stage][idx[stage][index]] = count
#     count = count + 1
#
# st_list = []
# et_list = []
# p_list = []
# job_id_list = []
# op_list = np.arange(cust_num)
# n_bay_start = []
#
# for i in range(stage_num):
#     for j in range(cust_num):
#         st_list.append(st[i][j])
#         et_list.append(et[i][j])
#         p_list.append(p_array[j, i])
#         job_id_list.append(j)
#         n_bay_start.append(machine_array[i][j])
#
# df = pd.DataFrame({'Machine': n_bay_start,
#                    'Start': st_list,
#                    'Finish': et_list,
#                    'Duration': p_list,
#                    'job_id': job_id_list})
# df.to_csv("gantt_data.csv")
