import tensorflow as tf
import os
def len_no_center(sample_solution):
    # 将sample_solution由python list转为tensor, sample_solution.shape=(sourceL,batch_size,input_dim)
    sample_solution = tf.stack(sample_solution, 0)
    # 将sample_solution后推一个位置, shape=(sourceL,batch_size,input_dim)
    sample_solution_tilted = tf.concat((tf.expand_dims(sample_solution[-1], 0), sample_solution[:-1]), 0)
    # dis_list: 路径各段距离, shape=(sourceL, batch_size)
    dis_list = tf.pow(tf.reduce_sum(tf.pow((sample_solution_tilted - sample_solution), 2), 2), .5)
    # sourceL: 路径段数
    sourceL, batch_size = dis_list.shape.as_list()
    # mask.shape=dis_list.shape=(sourceL, batch_size)
    # mask为二进制矩阵，0代表该边被扣除了
    # not_equal得到shape=(batch_size,input_dim)
    # reduce_any(axis=1)得到shape=(batch_size)
    # 下面这句话代表了有哪些batch的第0段需要保留或者扣除
    mask=tf.expand_dims(tf.cast(tf.logical_and(
            tf.reduce_any(tf.not_equal(sample_solution[0], sample_solution[-1]),axis=1),
            tf.reduce_any(tf.not_equal(sample_solution_tilted[0], sample_solution[-1]),axis=1)),dtype=tf.float32),0)
    for index in range(1,sourceL):
        # 这里concat, 将每个段都连接起来
        mask=tf.concat([mask, tf.expand_dims(tf.cast(tf.logical_and(
            tf.reduce_any(tf.not_equal(sample_solution[index], sample_solution[-1]),axis=1),
            tf.reduce_any(tf.not_equal(sample_solution_tilted[index], sample_solution[-1]),axis=1)),dtype=tf.float32),0)],axis=0)
        sess = tf.Session()
        # print(sess.run(mask))
        # print("***********************************************")
    route_lens_decoded=tf.reduce_sum(tf.multiply(dis_list,mask),axis=0)
    print(sess.run(route_lens_decoded))
    return route_lens_decoded

sample_solution = [[[1., 1, 1],
                    [2, 2, 2],
                    [1, 2, 3]],

                   [[3, 3, 3],
                    [0, 0, 0],
                    [4, 4, 4]],

                   [[0, 0, 0],
                    [4, 4, 4],
                    [0, 0, 0]],

                   [[4, 4, 4],
                    [6, 6, 6],
                    [1, 1, 1]],

                   [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]]

len_no_center(sample_solution)