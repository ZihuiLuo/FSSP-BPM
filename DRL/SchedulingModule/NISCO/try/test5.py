import tensorflow as tf
import time
# idx=[[[0, 3, 6, 9, 10, 10, 10, 10, 10, 10, 10],
#       [1, 4, 7, 10, 10, 10, 10, 10, 10, 10, 10],
#       [2, 5, 8, 10, 10, 10, 10, 10, 10, 10, 10]],
#
#      [[10, 1, 3, 5, 7, 9, 10, 10, 10, 10, 10],
#       [0, 2, 4, 6, 8, 10, 10, 10, 10, 10, 10]]]
idx = [[[0, 3, 4],
                [1, 4, 4],
                [4, 2, 2],
                [4, 1, 4],
                [4, 0, 4]],

               [[4, 4, 4],
                [4, 4, 4],
                [2, 4, 0],
                [4, 4, 4],
                [4, 4, 4]],

               [[4, 4, 4],
                [3, 4, 3],
                [4, 4, 4],
                [4, 4, 4],
                [4, 4, 1]]]

process_time = [[[2., 4.],
                         [3., 2.],
                         [4., 2.],
                         [5., 3.]],

                        [[2., 4.],
                         [3., 2.],
                         [6., 2.],
                         [2., 8.]],

                        [[3., 2.],
                         [6., 4.],
                         [4., 2.],
                         [2., 5.]]]
# process_time = [[[20., 21.],
#                  [5., 20.],
#                  [17., 22.],
#                  [8., 20.],
#                  [11., 13.],
#                  [12., 10.],
#                  [22., 12.],
#                  [9., 7.],
#                  [14., 8.],
#                  [6., 8.]]]

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


def update_start_end_time(idx, stage):
    """
    end_time: tensor of shape (stage_num, batch_size, cust_num)
    process_time: tensor of shape (batch_size, cust_num, stage_num)
    idx: cust_num selected, tensor of shape (depot_num, node_num, batch_size)
    stage: int
    """
    global node_num
    global cust_num
    global batch_size
    global depot_num
    global start_time
    global end_time
    global process_time
    for batch in range(batch_size):
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
                                    (end_time[stage - 1, batch, node_list[0] - 1] if stage != 0 else 0) +
                                    process_time_node[0])
            # 第i个结点的结束时间为（max（自身上一个阶段的结束时间，上一个结点的结束时间）+ 其process_time）
            for i in range(1, node_num):
                end_time_node = modify1(end_time_node, i,
                                        tf.reduce_max(
                                            [(end_time[stage - 1, batch, node_list[i] - 1] if stage != 0 else 0),
                                             end_time_node[i - 1]],
                                            axis=0) +
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


node_num = 5
cust_num = 4
stage_num = 2
batch_size = 3
depot_num = [3, 3]
start_time = tf.zeros((stage_num, batch_size, cust_num))
end_time = tf.zeros((stage_num, batch_size, cust_num))
process_time = tf.convert_to_tensor(process_time)
idx = tf.convert_to_tensor(idx)
t1 = time.time()
for stage in range(stage_num):
    update_start_end_time(idx, stage)
sess = tf.Session()
print("start time:")
print(sess.run(start_time))
print("end time:")
print(sess.run(end_time))
t2 = time.time()
print("time=",t2-t1)