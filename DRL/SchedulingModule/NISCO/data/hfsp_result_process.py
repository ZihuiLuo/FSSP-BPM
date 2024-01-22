import numpy as np
import pandas as pd
import tensorflow as tf


def seem_equal(a, b):
    return sum(abs(a - b)) < 1e-5


# 本方法将归一化input test data反正规化，还原并返回原始数据
def norm_inverse(path):
    columns = ['TOT_WGT', 'DEL_TO_HOURS', 'S1_PROCESS_TIME', 'S2_PROCESS_TIME','S3_PROCESS_TIME', 'S4_PROCESS_TIME']
    df = pd.DataFrame(path, columns=columns)
    # 反正规化
    without_norm = pd.read_csv("test_without_norm.csv")
    norm_inverse = lambda x, low, high: (x * (high - low) + low)
    low = {name: without_norm[name].min() for name in columns[1:]}
    high = {name: without_norm[name].max() for name in columns[1:]}
    new_df = {name: norm_inverse(df[name], low[name], high[name]) for name in columns[1:]}
    new_df = pd.DataFrame(new_df)
    new_df['TOT_WGT'] = df['TOT_WGT']
    order = ['TOT_WGT', 'DEL_TO_HOURS', 'S1_PROCESS_TIME', 'S2_PROCESS_TIME','S3_PROCESS_TIME', 'S4_PROCESS_TIME']
    new_df = new_df[order]
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

        for depot in range(depot_num[stage]):
            # 该配送中心选择的订单
            node_list = idx[depot, :, batch]
            # cust_num代表空结点转化为0代表空结点
            node_list = tf.map_fn(lambda node: tf.cond(tf.equal(node, cust_num), lambda: 0, lambda: node + 1),
                                  node_list, dtype=tf.int32)

            # 选择的订单对应的process_time
            process_time_node = tf.map_fn(lambda node: tf.cond(tf.equal(node, 0),
                                                               lambda: 0.,
                                                               lambda: process_time[batch, node - 1, stage]),
                                          node_list, dtype=tf.float32)
            # 选择的订单对应的结束时间，下标为node
            end_time_node = tf.zeros(node_num)
            # 第一个结点的结束时间为其上一个阶段的结束时间加其process_time
            end_time_node = modify1(end_time_node, 0,
                                    tf.cond(tf.equal(node_list[0], 0),
                                            lambda: 0.,
                                            lambda: tf.reduce_max(
                                                [(end_time[stage - 1, batch, node_list[0] - 1] if stage != 0 else 0.)
                                                    , 0.], axis=0))
                                    + process_time_node[0])
            # 第i个结点的结束时间为（max（自身上一个阶段的结束时间，上一个结点的结束时间）+ 其process_time）
            for i in range(1, node_num):
                end_time_node = modify1(end_time_node, i,
                                        tf.cond(tf.equal(node_list[i], 0),
                                                lambda: end_time_node[i - 1],
                                                lambda: tf.reduce_max(
                                                    [(end_time[
                                                          stage - 1, batch, node_list[i] - 1] if stage != 0 else 0.),
                                                     end_time_node[i - 1]],
                                                    axis=0)) +
                                        process_time_node[i])
            # 除去那些不在该配送中心的点
            node_not_zero = tf.map_fn(lambda node: tf.cast(tf.not_equal(node, 0), dtype=tf.float32),
                                      process_time_node, dtype=tf.float32)
            end_time_node = end_time_node * node_not_zero
            # 将结束时间写入end_time，减去process_time作为start_time
            for i in range(node_num):
                end_time = modify3(end_time, (stage, batch, node_list[i] - 1), end_time_node[i])
                start_time = modify3(start_time, (stage, batch, node_list[i] - 1),
                                     end_time_node[i] - process_time_node[i])

        batch = batch + 1
        return batch, n, end_time, start_time

    batch, n, end_time, start_time = tf.while_loop(cond, body, [batch, n, end_time, start_time])


# 测试时正规化的输入数据
list =[[490.015, 0.0, 0.6363636, 0.5744681, 0.59574467, 0.6], [496.979, 0.0, 0.34848484, 0.34042552, 0.31914893, 0.36], [97.302, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [484.543, 0.0, 0.34848484, 0.34042552, 0.31914893, 0.36], [494.253, 0.0, 0.121212125, 0.10638298, 0.10638298, 0.12], [489.256, 0.0, 0.54545456, 0.5319149, 0.5106383, 0.52], [497.5, 0.0, 1.0, 1.0, 1.0, 1.0], [495.225, 0.0, 0.57575756, 0.5531915, 0.5319149, 0.56], [491.218, 0.0, 0.18181819, 0.17021276, 0.17021276, 0.16], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

# 各配送中心各阶段作业加工顺序
idx=[[[2, 4, 8, 10, 10, 10, 10, 10, 10, 10, 10], [5, 1, 0, 10, 10, 10, 10, 10, 10, 10, 10], [9, 6, 7, 10, 10, 10, 10, 10, 10, 10, 10]], [[2, 4, 8, 10, 10, 10, 10, 10, 10, 10, 10], [5, 1, 0, 10, 10, 10, 10, 10, 10, 10, 10], [9, 6, 7, 10, 10, 10, 10, 10, 10, 10, 10]], [[2, 4, 8, 10, 10, 10, 10, 10, 10, 10, 10], [5, 1, 0, 10, 10, 10, 10, 10, 10, 10, 10], [9, 6, 7, 10, 10, 10, 10, 10, 10, 10, 10]], [[2, 5, 9, 4, 1, 6, 8, 0, 7, 10, 10]]]

node_num = 11
cust_num = 10
stage_num = 4
batch_size = 1
depot_num = [3,3,3,1]

array = np.array(list[:-1])
# origin_test_data为还原后的输入数据
origin_test_data = norm_inverse(array)
print(origin_test_data)
# process_time为测试数据的各阶段加工时间（目前时2阶段）
process_time = []
p_time = origin_test_data.iloc[:, 2:].values.tolist()
print("process_time:\n", p_time)
process_time.append(p_time)

start_time = tf.zeros((stage_num, batch_size, cust_num))
end_time = tf.zeros((stage_num, batch_size, cust_num))
process_time = tf.convert_to_tensor(process_time)


#idx = [[[0, 3, 6, 9, 10, 10, 10, 10, 10, 10, 10], [1, 4, 7, 10, 10, 10, 10, 10, 10, 10, 10], [2, 5, 8, 10, 10, 10, 10, 10, 10, 10, 10]], [[10, 1, 3, 5, 7, 9, 10, 10, 10, 10, 10], [0, 2, 4, 6, 8, 10, 10, 10, 10, 10, 10]]]


sess = tf.Session()
for stage in range(stage_num):
    stage_id = tf.convert_to_tensor(idx[stage])
    stage_id = tf.expand_dims(stage_id, -1)
    update_start_end_time1(stage_id, stage)

print("start_time:\n", sess.run(start_time))
print("end_time:\n", sess.run(end_time))

makespan = tf.reduce_max(end_time[-1],1)
print("makespan:",sess.run(makespan))

# 下面将start_time和end_time存入一维列表
st = np.round(np.squeeze(sess.run(start_time)))
et = np.round(np.squeeze(sess.run(end_time)))
p_array = np.round(np.array(p_time))
count = 0
# 这里解决batch_size=1的情况
machine_array = np.zeros((stage_num, cust_num))
for stage in range(stage_num):
    for depot in range(depot_num[stage]):
        for index in range(node_num):
            if (idx[stage][depot][index]) != cust_num:
                machine_array[stage][idx[stage][depot][index]] = count

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
