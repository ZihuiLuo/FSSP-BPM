import numpy as np
import tensorflow as tf


fname = "../data/train.txt"
depot_num = 3
node_num = 5
n_cust = 4
batch_size = 3
input_dim = 4
stage_num = 3
actions_temp = []
logprobs = []
probs = []
idxs = []
data = np.loadtxt(fname)
row_rand_array = np.arange(data.shape[0])
np.random.shuffle(row_rand_array)
data = data[row_rand_array[0:(batch_size * n_cust)]].reshape(batch_size, n_cust, input_dim)
depot = np.zeros((batch_size, 1, input_dim))
input_data = np.concatenate([data, depot], 1)
demand = input_data[:, :, 0]
mask = tf.zeros([stage_num, batch_size, node_num],
                     dtype=tf.float32)

def step(idx,stage):
    global demand,mask
    sess = tf.Session()
    BatchSequence = tf.expand_dims(tf.cast(tf.range(batch_size), tf.int64), 1)
    batched_idx = tf.concat([BatchSequence, idx], 1)
    d_sat = tf.gather_nd(demand, batched_idx)
    d_scatter = tf.scatter_nd(batched_idx, d_sat, tf.cast(tf.shape(demand), tf.int64))
    demand = tf.subtract(demand, d_scatter)
    this_mask = tf.expand_dims(tf.concat([tf.cast(tf.equal(demand, 0), tf.float32)[:, :-1],
                                          tf.zeros([batch_size, 1])], 1),0)


    if stage > 0: #如果不是第0阶段，就有pre_mask
        pre_mask = mask[:stage]
        if stage == stage_num - 1:  # 如果到最后一个阶段,就没有next_mask
            mask = tf.concat([pre_mask, this_mask], 0)
        else:  # 拼接3部分
            next_mask = mask[stage + 1:]
            mask = tf.concat([pre_mask, this_mask,next_mask], 0)
    else: #第0阶段，没有pre_mask
        next_mask = mask[stage + 1:]
        mask = tf.concat([this_mask, next_mask], 0)


def loop():
    global actions_temp

    BatchSequence = tf.expand_dims(tf.cast(tf.range(batch_size), tf.int64), 1)
    sess = tf.Session()
    stage_actions = []
    stage_id = []
    for stage in range(stage_num):

        for i in range(node_num):
            temp_actions=[]
            temp_idxs = []
            for j in range(depot_num):
                idx = np.random.randint(0,node_num,size=(batch_size,1))
                state = step(idx,stage)
                # batched_idx = tf.concat([BatchSequence, idx], 1)

        #         action = tf.gather_nd(tf.tile(input_data, [1, 1, 1]), batched_idx)
        #         temp_actions.append(action)
        #         idx = tf.expand_dims(idx, 1)
        #         temp_idxs.append(idx)
        #     stage_actions.append(temp_actions)
        #     stage_id.append((temp_idxs))
        # sess = tf.Session()
        # stage_actions = tf.transpose(stage_actions, perm=[1, 0, 2, 3])
        # stage_id = tf.transpose(tf.squeeze(stage_id,axis=4),perm=[1, 0, 2, 3])
        print("mask:\n", sess.run(tf.shape(mask)))
    #print(sess.run(tf.shape(stage_actions)))
    #print(sess.run(stage_id))
    #sess = tf.Session()
    # print(sess.run(tf.shape(actions_temp)))
    # print(sess.run(tf.shape(tf.transpose(actions_temp,perm=[1, 0, 2,3]))))
    # print(sess.run(tf.transpose(actions_temp,perm=[1, 0, 2,3])))

    # example_output=[]
    # for idx, action in enumerate(idxs):
    #     temp = []
    #     # example_output.append(list(action[R_ind0*np.shape(batch)[0]]))
    #     for i, ac in enumerate(action):
    #         temp.append(np.squeeze(ac).tolist())
    #     example_output.append(temp)
    # print(example_output)





loop()