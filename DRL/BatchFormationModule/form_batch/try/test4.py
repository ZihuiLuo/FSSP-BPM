import tensorflow as tf

def len_center(sample_solution):
	sample_solution = tf.stack(sample_solution,0)
	sample_solution_tilted = tf.concat((tf.expand_dims(sample_solution[-1],0),sample_solution[:-1]),0)
	# 计算每个batch的路线总长度
	route_lens_decoded = tf.reduce_sum(tf.pow(tf.reduce_sum(tf.pow((sample_solution_tilted - sample_solution) ,2), 2) , .5), 0)
	# route_lens_decoded的维度为 1* batch_size
	return route_lens_decoded


def len_no_center(sample_solution,sess):
	sample_solution = tf.stack(sample_solution, 0)
	sample_solution_tilted = tf.concat((tf.expand_dims(sample_solution[-1], 0), sample_solution[:-1]), 0)
	# dis_list: 路径各段距离
	dis_list=tf.pow(tf.reduce_sum(tf.pow((sample_solution_tilted - sample_solution), 2), 2), .5)
	sourceL,batch_size=tf.shape(dis_list).eval(session=sess)
	zeros=tf.Variable(batch_size*[0.],name="zeros")
	route_lens_decoded=tf.Variable(batch_size*[0.],name="route_lens_decoded")
	sess.run(tf.global_variables_initializer())
	for index in range(sourceL):
		condition=tf.logical_and(tf.reduce_any(tf.not_equal(sample_solution[index],sample_solution[-1])),
								 tf.reduce_any(tf.not_equal(sample_solution_tilted[index],sample_solution[-1])))
		route_lens_decoded=tf.add(route_lens_decoded,
								  tf.cond(condition,lambda:dis_list[index],lambda:zeros).eval(session=sess))
	# route_lens_decoded的维度为 1* batch_size
	return route_lens_decoded
def len_no_center2(sample_solution,sess):
	sample_solution = tf.stack(sample_solution, 0)
	tensor = tf.constant([1.0, 2.0, 3.0])
	# 将sample_solution后推一个位置, shape=(sourceL,batch_size,input_dim)
	sample_solution_tilted = tf.concat((tf.expand_dims(sample_solution[-1], 0), sample_solution[:-1]), 0)
	# dis_list: 路径各段距离, shape=(sourceL, batch_size)
	dis_list = tf.pow(tf.reduce_sum(tf.multiply(tf.pow((sample_solution_tilted - sample_solution), 2),tensor), 2), .5)
	sess = tf.Session()
	print("dis_list:\n", sess.run(dis_list))
	# sourceL: 路径段数
	sourceL, batch_size = dis_list.shape.as_list()
	# mask.shape=dis_list.shape=(sourceL, batch_size)
	# mask为二进制矩阵，0代表该边被扣除了
	# not_equal得到shape=(batch_size,input_dim)
	# reduce_any(axis=1)得到shape=(batch_size)
	# 下面这句话代表了有哪些batch的第0段需要保留或者扣除

	# 可考虑在reward里面加上划分炉次的个数，以及各批次剩余容量和（无委材数量）

	mask = tf.expand_dims(tf.cast(tf.logical_and(
		tf.reduce_any(tf.not_equal(sample_solution[0], sample_solution[-1]), axis=1),
		tf.reduce_any(tf.not_equal(sample_solution_tilted[0], sample_solution[-1]), axis=1)), dtype=tf.float32), 0)


	for index in range(1, sourceL):
		# 这里concat, 将每个段都连接起来
		mask = tf.concat([mask, tf.expand_dims(tf.cast(tf.logical_and(
			tf.reduce_any(tf.not_equal(sample_solution[index], sample_solution[-1]), axis=1),
			tf.reduce_any(tf.not_equal(sample_solution_tilted[index], sample_solution[-1]), axis=1)), dtype=tf.float32),
			0)], axis=0)
	print("mask:\n",sess.run(mask))
	route_lens_decoded = tf.reduce_sum(tf.multiply(dis_list, mask), axis=0) # 除去与配送中心之间剩下的距离
	print("route_lens_decoded:\n", sess.run(route_lens_decoded))
	# non_zeros_pos为dis_list中路径长度不为0的段（排除最后做完所有选择，停留在配送中心的惩罚）
	non_zeros_pos = tf.cast(tf.count_nonzero(tf.expand_dims(dis_list, axis=-1), axis=-1), dtype=tf.float32)
	print("non_zeros_pos:\n", sess.run(non_zeros_pos))
	# mask_flip为dis_list中路径长度不为0的段且是到配送中心的段
	mask_flip = tf.multiply(non_zeros_pos, (1 - mask) * 2)  # 返回配送中心惩罚为2（每段距离会被算2次
	print("mask_flip:\n",sess.run(mask_flip))
	route_lens_center = tf.reduce_sum(mask_flip, axis=0)
	print("route_lens_center:\n", sess.run(route_lens_center))
	route_lens_decoded += route_lens_center  # 一个批次内相邻距离的惩罚+回配送中心距离的惩罚
	##### reward = args['alpha']*route_lens_decoded + args['beta']*state[0]
	# state[0]是load，即当前卡车剩余容量，形状和route_lens_decoded一样

	return route_lens_decoded

sample_solution = [[[1., 1, 1],
					[4, 2, 1],
					[1, 2, 3]],

					[[3, 2, 3],
					[0, 0, 0],
					[4, 4, 4]],

					[[0, 0, 0],
					[3, 4, 4],
					[0, 0, 0]],

					[[4, 5, 4],
					[6, 5, 6],
					[1, 1, 1]],

					[[0, 0, 0],
					[0, 0, 0],
					[0, 0, 0]],

				   [[0, 0, 0],
					[0, 0, 0],
					[0, 0, 0]]]

sourceL = 6
batch_size = 3
input_dim = 3
sess=tf.Session()
print("len_center:\n",sess.run(len_center(sample_solution)))
print("len_no_center:\n",sess.run(len_no_center2(sample_solution,sess)))