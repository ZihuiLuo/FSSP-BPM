
import os
import numpy as np
import tensorflow as tf
import collections

originPath='C:\\Users\\cherry\\Desktop\\学习资料\\强化学习调研\\钢铁订单数据\\train.csv'
newPath='C:\\Users\\cherry\\Desktop\\学习资料\\强化学习调研\\钢铁订单数据\\test.txt'



# data = pd.read_csv(originPath, encoding='utf-8')
#
# def csv_to_txt():
#     if(os.path.exists(newPath)):
#         os.remove(newPath)
#
#     with open(newPath, 'a+', encoding='utf-8') as f:
#         for line in data.values:
#             f.write((str(line[1]) + '\t' + str(line[2]) +'\t'+str(line[3]) + '\t'+str(line[4]) + '\n'))
#
# def txt_to_matrix():
#
#     data = np.loadtxt(newPath)
#     print(data.shape)
#     # data = data.reshape(-1,3,4)
#     node_num = 5
#     batch_size =4
#
#     row_rand_array = np.arange(data.shape[0])
#     np.random.shuffle(row_rand_array)
#
#     data_new =data[row_rand_array[0:(batch_size * (node_num-1))]].reshape(batch_size,node_num-1,4)
#     depot = np.zeros((batch_size,1,4))
#     data = np.concatenate([data_new,depot],1)
#     print(data)
# txt_to_matrix()

fname = r"..\data\test.txt"
node = 6
batch_size = 3
if os.path.exists(fname):
    data = np.loadtxt(fname)
    row_rand_array = np.arange(data.shape[0])
    np.random.shuffle(row_rand_array)
    data = data[row_rand_array[0:(batch_size * (node-1))]].reshape(batch_size, (node-1), 4)
    depot = np.zeros((batch_size, 1, 4))
    input_data = np.concatenate([data, depot], 1)
    print("input_data\n",input_data)
else:
    print("error")
    input_data=[]
input_pnt = input_data[:,:,:3]
print("input_pnt\n",input_data)
demand = input_data[:,:,-1]
print("demand\n",demand)



demand2 = tf.tile(demand, [1,1])
load = tf.ones([batch_size])*50
mask = tf.zeros([batch_size*1,node],
                dtype=tf.float32)
mask2 = tf.concat([tf.cast(tf.equal(demand2,0), tf.float32)[:,:-1],
            tf.ones([batch_size,1])],1)

idx=[]
idx = tf.cast((node-1)*tf.ones([batch_size,1]),tf.int64)
BatchSequence = tf.expand_dims(tf.cast(tf.range(batch_size), tf.int64), 1)
batched_idx = tf.concat([BatchSequence, idx], 1)
tf_gather=tf.gather_nd(demand2,batched_idx)

d_sat = tf.minimum(tf.gather_nd(demand2,batched_idx), tf.cast(load,tf.float64))
d_scatter = tf.scatter_nd(batched_idx, d_sat, tf.cast(tf.shape(demand),tf.int64))
load_flag = tf.squeeze(tf.cast(tf.equal(idx, node-1), tf.float32), 1)

sess = tf.Session()
print("idx:\n",sess.run(idx))
print("batch_size\n",sess.run(tf.cast(tf.range(batch_size), tf.int64)))
print("BatchSequencex:\n",sess.run(BatchSequence))
print("batched_idx:\n",sess.run(batched_idx))
print("tf_gather:\n",sess.run(tf_gather))
print("d_sat:\n",sess.run(d_sat))
print("d_scatter\n",sess.run(d_scatter))
demand = tf.subtract(demand, d_scatter)
print("new_demand:\n",sess.run(demand))

print(sess.run(tf.cast(tf.equal(idx, node-1), tf.float32)))
print("load_flag\n",sess.run(load_flag))

print("load\n",sess.run(load))

class State(collections.namedtuple("State",
                                        ("load",
                                         "demand",
                                         'd_sat',
                                         "mask"))):
    pass


state = State(load=load,
              demand=demand,
              d_sat=tf.zeros([batch_size, node]),
              mask=mask)
print(sess.run(state[0]))

# print(sess.run(tf.equal(demand2,0)))
# print(sess.run(tf.cast(tf.equal(demand2,0), tf.float32)))
# print(sess.run(tf.cast(tf.equal(demand2,0), tf.float32)[:,:-1]))


#print("demand2\n",sess.run(demand2))
# print("load\n",sess.run(load))
# print("mask\n",sess.run(mask))
# print("mask2\n",sess.run(mask2))




