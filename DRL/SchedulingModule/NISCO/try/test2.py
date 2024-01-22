
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import collections
class State(collections.namedtuple("State",
                                   (
                                    "demand",
                                    "d_sat",
                                    "mask",
                                    "start_time",
                                    "end_time"))):
    pass
class Env(object):
    def __init__(self):

        self.node_num = 5
        self.n_cust = 4
        self.batch_size = 3
        self.n_problems = 2
        self.input_dim = 4
        self.stage_num = 2
        self.capacity = 100000
        self.depot_num = [3,3]

        self.train_data = np.loadtxt("../data/train.txt")
        sess = tf.Session()
        row_rand_array = np.arange(self.train_data.shape[0])
        np.random.shuffle(row_rand_array)
        data_without_norm = self.train_data[row_rand_array[0:(self.batch_size * self.n_cust)]]
        norm = lambda x: (x - x.min()) / (x.max() - x.min())
        data_without_norm = pd.DataFrame(data_without_norm,
                                         columns=['TOT_WGT', 'DEL_TO_HOURS', 'S1_PROCESS_TIME', 'S2_PROCESS_TIME'])
        data_without_norm['DEL_TO_HOURS'] = data_without_norm[['DEL_TO_HOURS']].apply(norm)
        data_without_norm['S1_PROCESS_TIME'] = data_without_norm[['S1_PROCESS_TIME']].apply(norm)
        data_without_norm['S2_PROCESS_TIME'] = data_without_norm[['S2_PROCESS_TIME']].apply(norm)
        data_new = np.array(data_without_norm.values.tolist()).reshape(self.batch_size, self.n_cust,
                                                                       self.input_dim)
        depot = np.zeros((self.batch_size, 1, self.input_dim))
        self.input_data = np.concatenate([data_new, depot], 1)
        #print("test_data_shape:\n",self.input_data[0:1].shape)
        self.demand = self.input_data[:, :, 0]
        self.demand = tf.expand_dims(self.demand, 2)
        print("shape of demand:",sess.run(tf.shape(self.demand)))

        print("max_time:",sess.run(tf.shape(self.demand[1])))
        #self.process_time = self.input_data[:, :, 2:] # (batch_size * nodes * stage_num)
        self.ddl = self.input_data[:,:,1][:,:self.n_cust]
        self.mask = tf.zeros([self.stage_num,self.batch_size,self.node_num],
                             dtype=tf.float32)
        self.start_time = tf.zeros([self.stage_num, self.batch_size, self.n_cust],
                             dtype=tf.float32)
        self.end_time = tf.zeros([self.stage_num, self.batch_size, self.n_cust],
                             dtype=tf.float32)
        self.d_sat = tf.zeros([self.stage_num, self.batch_size, self.n_cust],
                               dtype=tf.float32)
        #self.end_time = tf.concat([tf.ones([1,self.batch_size, self.n_cust]),tf.zeros([1,self.batch_size, self.n_cust]),tf.zeros([1,self.batch_size, self.n_cust])],0)

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
        self.process_time = tf.convert_to_tensor(process_time)
        print("shape of process_time:",tf.shape(self.process_time))
        # depot_num * node_num * batch_size
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
        self.idx = tf.cast(tf.transpose(tf.convert_to_tensor(idx),perm=[1,0,2]),dtype=tf.int64)

    def each_step_reset(self):  # 每个训练step都要初始化一次（新的input_data）
        '''
        Resets the environment. This environment might be used with different decoders.
        In case of using with beam-search decoder, we need to have to increase
        the rows of the mask by a factor of beam_width.
        '''
        state = State(
                      demand=self.demand,
                      d_sat=self.d_sat,
                      mask=self.mask,
                      start_time=self.start_time,
                      end_time=self.end_time)
        return state

    def each_stage_reset(self):
        self.demand = self.input_data[:, :, 0]



        self.load = tf.ones([self.batch_size]) * self.capacity

    def step(self,idx,stage):
        sess = tf.Session()
        self.stage = stage
        BatchSequence = tf.expand_dims(tf.cast(tf.range(self.batch_size), tf.int64), 1)
        batched_idx = tf.concat([BatchSequence, idx], 1)
        d_sat = tf.gather_nd(self.demand, batched_idx)
        d_scatter = tf.scatter_nd(batched_idx, d_sat, tf.cast(tf.shape(self.demand), tf.int64))
        self.demand = tf.subtract(self.demand, d_scatter)
        self.this_mask = tf.expand_dims(tf.concat([tf.cast(tf.equal(self.demand, 0), tf.float32)[:, :-1],
                                              tf.zeros([self.batch_size, 1])], 1), 0)
        #print("shape:this_mask:",sess.run(tf.shape(self.this_mask)))
        def first_stage():
            self.next_mask = self.mask[self.stage + 1:]
            return tf.concat([self.this_mask, self.next_mask], 0)

        def other_stage():
            self.pre_mask = self.mask[:self.stage]
            return tf.cond(tf.equal(self.stage,self.stage_num - 1), concat_first_two_parts, concat_three_parts)

        def concat_first_two_parts():
            return tf.concat([self.pre_mask, self.this_mask], 0)

        def concat_three_parts():
            self.next_mask = self.mask[self.stage + 1:]
            return tf.concat([self.pre_mask, self.this_mask, self.next_mask], 0)

        self.mask = tf.cond(tf.cast(self.stage>0,tf.bool), other_stage, first_stage)

    def loop(self):
        sess = tf.Session()
        actions_total = []
        idxs_total = []
        R_stage = tf.zeros((self.batch_size))
        print(sess.run(R_stage))
        BatchSequence = tf.expand_dims(tf.cast(tf.range(self.batch_size), tf.int64), 1)
        state = self.each_step_reset()
        #print("batchSequence:\n",sess.run(BatchSequence))
        for stage in range(self.stage_num):
            self.each_stage_reset()
            stage_idxs=[]
            stage_actions=[]
            #print("stage=",stage)
            for i in range(self.node_num):
                temp_idx=[]
                temp_action = []
                for j in range(self.depot_num[stage]):
                    idx = tf.transpose(tf.expand_dims(self.idx[i][j],0))
                    self.step(idx, stage)
                    batched_idx = tf.concat([BatchSequence, idx], 1)
                    action = tf.gather_nd(tf.tile(self.input_data, [1, 1, 1]), batched_idx)

                    temp_idx.append(idx)
                    temp_action.append(action)
                    #print(sess.run(temp_idx))
                stage_idxs.append(temp_idx)
                stage_actions.append(temp_action)
            stage_actions = tf.transpose(stage_actions,
                                         perm=[1, 0, 2, 3])  # （depot_num * node_num * batch_size*input_dim）
            R_stage = R_stage+self.reward_func_stage(stage_actions, stage)
            print("R_stage", sess.run(R_stage))
            stage_idxs = tf.squeeze(tf.transpose(stage_idxs,
                                     perm=[1, 0, 2, 3]),3)  # （depot_num * node_num * batch_size*1）
            state = self.update_start_end_time(stage_idxs,stage)

            actions_total.append(stage_actions)
            idxs_total.append(stage_idxs)

        R_all = self.reward_func(state)
        R = R_all+R_stage

        sess = tf.Session()
        print("reward=",sess.run(R))
        # print("process_time:\n",sess.run(self.process_time))
        print("start_time:\n", sess.run(self.start_time))
        print("end_time:\n", sess.run(self.end_time))

    def update_start_end_time(self, idx, stage):

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
        n = self.batch_size
        end_time = self.end_time
        start_time = self.start_time

        def cond(batch, n, end_time, start_time):
            return batch < n

        def body(batch, n, end_time, start_time):

            for depot in range(self.depot_num[stage]):
                # 该配送中心选择的订单
                node_list = idx[depot, :, batch]
                # cust_num代表空结点转化为0代表空结点
                node_list = tf.map_fn(lambda node: tf.cond(tf.equal(node, self.n_cust), lambda: 0,
                                                           lambda: tf.cast(node + 1, dtype=tf.int32)),
                                      node_list, dtype=tf.int32)

                # 选择的订单对应的process_time
                process_time_node = tf.map_fn(lambda node: tf.cond(tf.equal(node, 0),
                                                                   lambda: 0.,
                                                                   lambda: self.process_time[batch, node - 1, stage]),
                                              node_list, dtype=tf.float32)
                # 选择的订单对应的结束时间，下标为node
                end_time_node = tf.zeros(self.node_num)
                # 第一个结点的结束时间为其上一个阶段的结束时间加其process_time
                end_time_node = modify1(end_time_node, 0,
                                        tf.cond(tf.equal(node_list[0], 0),
                                                lambda: 0.,
                                                lambda: tf.reduce_max(
                                                    [(end_time[
                                                          stage - 1, batch, node_list[0] - 1] if stage != 0 else 0.)
                                                        , 0.], axis=0))
                                        + process_time_node[0])
                # 第i个结点的结束时间为（max（自身上一个阶段的结束时间，上一个结点的结束时间）+ 其process_time）
                for i in range(1, self.node_num):
                    end_time_node = modify1(end_time_node, i,
                                            tf.cond(tf.equal(node_list[i], 0),
                                                    lambda: end_time_node[i - 1],
                                                    lambda: tf.reduce_max(
                                                        [(end_time[
                                                              stage - 1, batch, node_list[
                                                                  i] - 1] if stage != 0 else 0.),
                                                         end_time_node[i - 1]],
                                                        axis=0)) +
                                            process_time_node[i])
                # 除去那些不在该配送中心的点
                node_not_zero = tf.map_fn(lambda node: tf.cast(tf.not_equal(node, 0), dtype=tf.float32),
                                          process_time_node, dtype=tf.float32)
                end_time_node = end_time_node * node_not_zero
                # 将结束时间写入end_time，减去process_time作为start_time
                for i in range(self.node_num):
                    end_time = modify3(end_time, (stage, batch, node_list[i] - 1), end_time_node[i])
                    start_time = modify3(start_time, (stage, batch, node_list[i] - 1),
                                         end_time_node[i] - process_time_node[i])

            batch = batch + 1
            return batch, n, end_time, start_time

        batch, n, end_time, start_time = tf.while_loop(cond, body, [batch, n, end_time, start_time])
        self.start_time = start_time
        self.end_time = end_time

        state = State(
                      demand=self.demand,
                      d_sat=tf.zeros([self.batch_size, self.node_num]),
                      mask=self.mask,
                      start_time=self.start_time,
                      end_time=self.end_time)
        return state



    def reward_func_stage(self,sample_solution,stage):
        sess = tf.Session()
        print("sample_solution_o.shape:", sess.run(tf.shape(sample_solution)))
        # python list to Tensor
        sample_solution = tf.stack(sample_solution, 0)
        # drop first demand column
        sample_solution = sample_solution[:, :, :, 2+stage]
        #print("sample_solution:",sess.run(sample_solution))
        print("sample_solution.shape:", sess.run(tf.shape(sample_solution)))
        # sum by node_num
        sumed = tf.reduce_sum(sample_solution, axis=1)
        # varience by depot_num
        mean, var = tf.nn.moments(sumed, axes=[0])
        # sum of deadline date varience and processing time varience

        print("mean",sess.run(mean))
        print("var", sess.run(var))
        # varience_penalty = tf.reduce_sum(var, axis=1)
        return  tf.cast(var, dtype=tf.float32)

    def reward_func(self,state): #最小makespan
        sess = tf.Session()
        mask = tf.stack(state[2], 0)
        superBigValue = 10000
        best_mask = tf.concat([tf.ones((self.stage_num,self.batch_size, self.node_num - 1)), tf.zeros((self.stage_num,self.batch_size, 1))], axis=2)
        mask_penalty = tf.reduce_sum(tf.reduce_sum((best_mask-mask),axis=2),axis=0)*superBigValue
        makespan = tf.reduce_max(self.end_time[-1],1)
        print("makespan",sess.run(makespan))
        return makespan + mask_penalty

    def update_start_end_time1(self, idx, stage):
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
        n = self.batch_size
        end_time = self.end_time
        start_time = self.start_time

        def cond(batch, n,end_time,start_time):
            return batch < n

        def body(batch, n,end_time,start_time):


            for depot in range(self.depot_num[stage]):
                # 该配送中心选择的订单
                node_list = idx[depot, :, batch]
                # cust_num代表空结点转化为0代表空结点
                node_list = tf.map_fn(lambda node: tf.cond(tf.equal(node, self.n_cust), lambda: 0,
                                                           lambda: tf.cast(node + 1, dtype=tf.int32)),
                                      node_list, dtype=tf.int32)
                # 选择的订单对应的process_time
                process_time_node = tf.map_fn(lambda node: tf.cond(tf.equal(node, 0),
                                                                   lambda: 0.,
                                                                   lambda: self.process_time[batch, node - 1, stage]),
                                              node_list, dtype=tf.float32)
                # 选择的订单对应的结束时间，下标为node
                end_time_node = tf.zeros(self.node_num)
                # 第一个结点的结束时间为其上一个阶段的结束时间加其process_time
                end_time_node = modify1(end_time_node, 0,
                                        (end_time[stage - 1, batch, node_list[0] - 1] if stage != 0 else 0) +
                                        process_time_node[0])
                # 第i个结点的结束时间为（max（自身上一个阶段的结束时间，上一个结点的结束时间）+ 其process_time）
                for i in range(1, self.node_num):
                    end_time_node = modify1(end_time_node, i,
                                            tf.reduce_max(
                                                [(end_time[
                                                      stage - 1, batch, node_list[i] - 1] if stage != 0 else 0),
                                                 end_time_node[i - 1]],
                                                axis=0) +
                                            process_time_node[i])
                # 除去那些不在该配送中心的点
                node_not_zero = tf.map_fn(lambda node: tf.cast(tf.not_equal(node, 0), dtype=tf.float32),
                                          process_time_node, dtype=tf.float32)
                end_time_node = end_time_node * node_not_zero
                # 将结束时间写入end_time，减去process_time作为start_time
                for i in range(self.node_num):
                    end_time = modify3(end_time, (stage, batch, node_list[i] - 1), end_time_node[i])
                    start_time = modify3(start_time, (stage, batch, node_list[i] - 1),
                                              end_time_node[i] - process_time_node[i])

            batch = batch + 1
            return batch, n,end_time,start_time

        batch,n,end_time,start_time = tf.while_loop(cond, body, [batch, n,end_time,start_time])
        self.start_time = start_time
        self.end_time = end_time

        state = State(
            demand=self.demand,
            d_sat=self.d_sat,
            mask=self.mask,
            start_time=start_time,
            end_time=end_time)
        return state


env = Env()
env.loop()


# stage_num * depot_num[stage] * node_num * batch_size * input_dim