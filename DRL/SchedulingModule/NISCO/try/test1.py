import tensorflow as tf

def reward_func(sample_solution,mask):
    '''
    sample_solution比之前新增depot维度,node_num包括了配送中心
    sample_solution.shape=(depot_num,node_num,batch_size,input_dim)
    mask.shape=(batch_size,node_num)
    '''
    superBigValue = 10000000
    # python list to Tensor
    sample_solution=tf.stack(sample_solution, 0)
    mask = tf.stack(mask, 0)
    sess = tf.Session()
    # get shape
    depot_num, node_num, batch_size, input_dim = sample_solution.shape
    '''
    best_mask = [[1., 1., 1., 0.],
                 [1., 1., 1., 0.]]
    '''
    best_mask = tf.concat([tf.ones((batch_size, node_num-1)), tf.zeros((batch_size,1))],axis=1)
    mask_penalty=tf.reduce_sum(best_mask-mask,axis=1)*superBigValue
    print("mask_penalty:\n",sess.run(mask_penalty))
    # drop first demand column
    sample_solution = sample_solution[:, :, :, 1:]

    print("sample_solution:\n",sess.run(sample_solution))
    # sum by node_num
    sumed = tf.reduce_sum(sample_solution,axis=1)
    print("sumed:\n", sess.run(sumed))
    # varience by depot_num
    mean, var = tf.nn.moments(sumed, axes=[0])
    # sum of deadline date varience and processing time varience

    print("var:\n",sess.run(var))
    varience_penalty=tf.reduce_sum(var,axis=1)
    return mask_penalty+varience_penalty
    #return varience_penalty

sess = tf.InteractiveSession()
sample_solution = \
[[[[4.2110000e+00, 2.0160615e+07, 9.0000000e+00],
   [6.2100000e+00, 2.0151215e+07, 1.0000000e+00]],

  [[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
   [6.2100000e+00, 2.0151215e+07, 1.0000000e+00]],

  [[2.6100000e+00, 2.0151031e+07, 8.0000000e+00],
   [0.0000000e+00, 0.0000000e+00, 0.0000000e+00]],

  [[4.438000e+00, 2.018043e+07, 3.000000e+00],
   [0.000000e+00, 0.000000e+00, 0.000000e+00]]],

 [[[4.4380000e+00, 2.0180430e+07, 3.0000000e+00],
   [6.4160000e+00, 2.0150115e+07, 4.0000000e+00]],

  [[4.2110000e+00, 2.0160615e+07, 9.0000000e+00],
   [6.4160000e+00, 2.0150115e+07, 4.0000000e+00]],

  [[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
   [6.2100000e+00, 2.0151215e+07, 1.0000000e+00]],

  [[0.000000e+00, 0.000000e+00, 0.000000e+00],
   [3.957000e+00, 2.015113e+07, 1.000000e+00]]],

 [[[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
   [6.4160000e+00, 2.0150115e+07, 4.0000000e+00]],

  [[2.6100000e+00, 2.0151031e+07, 8.0000000e+00],
   [3.9570000e+00, 2.0151130e+07, 1.0000000e+00]],

  [[4.2110000e+00, 2.0160615e+07, 9.0000000e+00],
   [0.0000000e+00, 0.0000000e+00, 0.0000000e+00]],

  [[2.6100000e+00, 2.0151031e+07, 8.0000000e+00],
   [6.2100000e+00, 2.0151215e+07, 1.0000000e+00]]]]

mask = [[0., 0., 1., 0.],
        [0., 1., 0., 0.]]
print(sess.run(reward_func(sample_solution,mask)))
