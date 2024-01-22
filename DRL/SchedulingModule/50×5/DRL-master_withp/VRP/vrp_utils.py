import numpy as np
import tensorflow as tf
import os
import warnings
import collections
import pandas as pd

def load_VRP_dataset(args):
    n_problems, n_cust, data_dir = args['test_size'], args['n_cust'], args['data_dir']
    input_dim = args['input_dim']
    print('load test data.')
    fname = os.path.join(data_dir, "test.txt")

    # load test data
    if os.path.exists(fname):
        data = np.loadtxt(fname)

        # if len(data) < n_cust:
        #     append_len = n_cust - len(data)
        #     zeros = np.zeros((append_len, input_dim))
        #     data = np.concatenate([data, zeros])
        #     np.savetxt(fname, data, fmt='%f', delimiter='\t')

        # row_rand_array = np.arange(data.shape[0])
        # np.random.shuffle(row_rand_array)
        # data = data[row_rand_array[0:(n_problems * n_cust)]].reshape(n_problems, n_cust, input_dim)
        # depot = np.zeros((n_problems, 1, input_dim))
        # input_data = np.concatenate([data, depot], 1)
    else:
        print('no test data!')
    return data


class DataGenerator(object):
    def __init__(self, args):

        self.args = args
        # create test data
        self.n_problems = args['test_size']
        self.input_dim = args['input_dim']
        self.test_data = load_VRP_dataset(args)
        self.train_data = np.loadtxt(os.path.join(self.args['data_dir'], "train.txt"))
        self.test_num = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.test_num = 0
        self.count = 0
        self.shuffle_data()

    def shuffle_data(self):
        # shuffle_data在每个epoch开始都执行
        row_rand_array = np.arange(self.train_data.shape[0])
        np.random.shuffle(row_rand_array)
        self.train_data = self.train_data[row_rand_array]

    def get_train_next(self):
        '''
        Get next batch of problems for training
        Retuens:
            input_data: data with shape [batch_size x max_time x input_dim]
        '''
        # stage_num = self.args['stage_num']
        # order = ['demand']
        # p = ['p' + str(i + 1) for i in range(stage_num)]
        # order.extend(p)
        #
        # row_rand_array = np.arange(self.train_data.shape[0])
        # np.random.shuffle(row_rand_array)
        # data_without_norm = self.train_data[row_rand_array[0:(self.args['batch_size'] * self.args['n_cust'])]]
        # norm = lambda x: (x - x.min()) / (x.max() - x.min())
        # data_without_norm = pd.DataFrame(data_without_norm,
        #                                  columns=order)
        # train_with_norm = data_without_norm[p].apply(norm)
        # train_with_norm['demand'] = data_without_norm['demand']
        # data_new = np.array(train_with_norm.values.tolist()).reshape(self.args['batch_size'], self.args['n_cust'],
        #                                                       self.input_dim)
        # depot = np.zeros((self.args['batch_size'], 1, self.input_dim))
        # input_data = np.concatenate([data_new, depot], 1)

        #return input_data
        # load train data
        train_batch = self.train_data[self.count:(self.count + self.args['batch_size'] * self.args['n_cust'])]
        self.count = self.count + self.args['batch_size'] * self.args['n_cust']

        # cust_data = np.array(train_batch.values.tolist()).reshape(self.args['batch_size'], self.args['n_cust'], self.input_dim)
        cust_data = np.array(train_batch).reshape(self.args['batch_size'], self.args['n_cust'],
                                                  self.input_dim)
        depot = np.zeros((self.args['batch_size'], 1, self.input_dim))
        batch_data = np.concatenate([cust_data, depot], 1)

        return batch_data


    def get_test_next(self):
        '''
        Get next batch of problems for testing
        '''
        test_batch = self.test_data[self.test_num:(self.test_num + self.args['n_cust'])]
        self.test_num = self.test_num + self.args['n_cust']

        # cust_data = np.array(train_batch.values.tolist()).reshape(self.args['batch_size'], self.args['n_cust'], self.input_dim)
        cust_data = np.array(test_batch).reshape(1, self.args['n_cust'],
                                                  self.input_dim)
        depot = np.zeros((1, 1, self.input_dim))
        batch_data = np.concatenate([cust_data, depot], 1)

        return batch_data

    def get_test_all(self):
        '''
        Get all test problems
        '''
        return self.test_data

class State(collections.namedtuple("State",
                                   ("load",
                                    "demand",
                                    "d_sat",
                                    "mask",
                                    #"total_mask",
                                    "start_time",
                                    "end_time"))):
    pass


class Env(object):
    def __init__(self,
                 args):
        '''
        This is the environment for VRP.
        Inputs:
            args: the parameter dictionary. It should include:
                args['n_nodes']: number of nodes in VRP
                args['n_custs']: number of customers in VRP
                args['input_dim']: dimension of the problem which is 3
        '''
        self.capacity = args['capacity']
        self.n_nodes = args['n_nodes']
        self.n_cust = args['n_cust']
        self.input_dim = args['input_dim']
        self.depot_num = args['depot_num'] #各个阶段的机器数量
        self.stage_num = args['stage_num'] # 一共几个阶段
        self.stage = 0


        self.input_data = tf.placeholder(tf.float32, \
                                         shape=[None, self.n_nodes, self.input_dim])

        #这里input_pnt为输入的所有数据
        self.input_pnt = self.input_data
        self.demand = self.input_data[:, :, 0]
        self.process_time = self.input_data[:,:,1:] # 各个阶段作业加工时间
        self.batch_size = tf.shape(self.input_pnt)[0]  # 批次大小
        # self.total_mask = tf.zeros([self.stage_num, self.batch_size, self.n_nodes],
        #                      dtype=tf.float32)
        # self.total_mask = tf.concat(
        #     [tf.zeros((self.stage_num, self.batch_size, self.n_nodes - 1)), tf.ones((self.stage_num, self.batch_size, 1))],
        #     axis=2)
        self.start_time = tf.zeros([self.stage_num, self.batch_size, self.n_nodes],
                                   dtype=tf.float32)
        self.end_time = tf.zeros([self.stage_num, self.batch_size, self.n_nodes],
                                 dtype=tf.float32)
        self.stage_endtime = self.end_time[self.stage]


    def each_step_reset(self, beam_width=1):# 每个训练step都要初始化一次（新的input_data）
        '''
        Resets the environment. This environment might be used with different decoders.
        In case of using with beam-search decoder, we need to have to increase
        the rows of the mask by a factor of beam_width.
        '''

        # dimensions
        self.beam_width = beam_width
        self.batch_beam = self.batch_size * beam_width
        self.input_pnt = self.input_data
        self.demand = self.input_data[:, :, 0]
        # demand: [batch_size * beam_width, max_time]
        self.demand = tf.tile(self.demand, [self.beam_width, 1])
        self.process_time = self.input_data[:, :, 1:]
        # load: [batch_size * beam_width]
        self.load = tf.ones([self.batch_beam]) * self.capacity
        # create mask,初始化全为0
        self.mask = tf.concat(
            [tf.zeros((self.batch_size * beam_width, self.n_nodes - 1)), tf.ones((self.batch_size * beam_width, 1))],
            axis=1)

        # total_mask 的形状为:(stage_num * batch_size * n_nodes)
        # self.total_mask = tf.concat(
        #     [tf.zeros((self.stage_num, self.batch_size, self.n_nodes - 1)), tf.ones((self.stage_num, self.batch_size, 1))],
        #     axis=2)
        self.start_time = tf.zeros([self.stage_num, self.batch_size, self.n_nodes],
                                   dtype=tf.float32)
        self.end_time = tf.zeros([self.stage_num, self.batch_size, self.n_nodes],
                                 dtype=tf.float32)

    def each_stage_reset(self,stage):
        '''
        每阶段开始要调用一次，主要是将每个订单的demand的值改为初始demand,每阶段的mask要重置
        注：每阶段的total_mask没用重置，只有每个step开始才有重置total_mask
        '''
        self.load = tf.ones([self.batch_size]) * self.capacity
        # self.can_start = tf.zeros([1, self.batch_size, self.n_cust],
        #                           dtype=tf.float32)
        self.demand = self.input_data[:, :, 0]
        # self.mask = tf.zeros([self.batch_size, self.n_nodes],
        #                      dtype=tf.float32)
        self.mask = tf.concat(
            [tf.zeros((self.batch_size, self.n_nodes - 1)), tf.ones((self.batch_size, 1))],
            axis=1)
        self.stage = stage
        self.stage_endtime = self.end_time[self.stage]


    def update_start_end_time(self,idx,stage):

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

        # （node_num * batch_size）

        def cond(batch, n, end_time, start_time):
            return batch < n

        def body(batch, n, end_time, start_time):
            # 该配送中心选择的订单
            node_list = idx[:, batch]
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
            end_time_node = tf.zeros(self.n_nodes-1 )
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
            for i in range(1, self.n_nodes-1):
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
            for i in range(self.n_nodes-1):
                end_time = modify3(end_time, (stage, batch, node_list[i] - 1), end_time_node[i])
                start_time = modify3(start_time, (stage, batch, node_list[i] - 1),
                                         end_time_node[i] - process_time_node[i])
            # 该配送中心选择的订单
            # node_list = idx[:, batch]
            #
            # node_list = tf.map_fn(
            #     lambda node: tf.cast(node, dtype=tf.int32),
            #     node_list, dtype=tf.int32)
            #
            #
            # process_time_node = tf.map_fn(lambda node: self.process_time[batch, node, stage],
            #                               node_list, dtype=tf.float32)
            # # 选择的订单对应的结束时间，下标为node
            # end_time_node = tf.zeros(self.n_nodes - 1)
            # # 第一个结点的结束时间为其上一个阶段的结束时间加其process_time
            #
            # end_time_node = modify1(end_time_node, 0,
            #                         tf.reduce_max(
            #                                     [(end_time[
            #                                           stage - 1, batch, node_list[0]] if stage != 0 else 0.)
            #                                         , 0.], axis=0)
            #                         + process_time_node[0])
            # # 第i个结点的结束时间为（max（自身上一个阶段的结束时间，上一个结点的结束时间）+ 其process_time）
            # for i in range(1, self.n_nodes - 1):
            #     end_time_node = modify1(end_time_node, i,
            #                             tf.reduce_max(
            #                                         [(end_time[
            #                                               stage - 1, batch, node_list[
            #                                                   i]] if stage != 0 else 0.),
            #                                          end_time_node[i - 1]],
            #                                         axis=0) +
            #                             process_time_node[i])
            #
            # # 将结束时间写入end_time，减去process_time作为start_time
            # for i in range(self.n_nodes - 1):
            #     end_time = modify3(end_time, (stage, batch, node_list[i]), end_time_node[i])
            #     start_time = modify3(start_time, (stage, batch, node_list[i]),
            #                          end_time_node[i] - process_time_node[i])


            batch = batch + 1
            return batch, n, end_time, start_time

        batch, n, end_time, start_time = tf.while_loop(cond, body, [batch, n, end_time, start_time])
        self.start_time = start_time
        self.end_time = end_time
        self.stage_endtime = self.end_time[self.stage]


        state = State(load=self.load,
                      demand=self.demand,
                      d_sat=tf.zeros([self.batch_beam, self.n_nodes]),
                      mask=self.mask,
                      #total_mask=self.total_mask,
                      start_time=self.start_time,
                      end_time=self.end_time)
        return state


    def step(self,
             idx,stage,
             beam_parent=None):
        '''
        runs one step of the environment and updates demands, loads and masks
        '''

        # if the environment is used in beam search decoder
        if beam_parent is not None:
            # BatchBeamSeq: [batch_size*beam_width x 1]
            # [0,1,2,3,...,127,0,1,...],
            batchBeamSeq = tf.expand_dims(tf.tile(tf.cast(tf.range(self.batch_size), tf.int64),
                                                  [self.beam_width]), 1)
            # batchedBeamIdx:[batch_size*beam_width]
            batchedBeamIdx = batchBeamSeq + tf.cast(self.batch_size, tf.int64) * beam_parent
            # demand:[batch_size*beam_width x sourceL]
            self.demand = tf.gather_nd(self.demand, batchedBeamIdx)
            # load:[batch_size*beam_width]
            self.load = tf.gather_nd(self.load, batchedBeamIdx)
            # MASK:[batch_size*beam_width x sourceL]
            self.mask = tf.gather_nd(self.mask, batchedBeamIdx)

        BatchSequence = tf.expand_dims(tf.cast(tf.range(self.batch_beam), tf.int64), 1)
        batched_idx = tf.concat([BatchSequence, idx], 1)

        # 假设batch_beam=3，n_nodes=6
        # 初始idx=[[5]
        #          [5]
        #          [5]]
        # BatchSequencex = [[0]
        #                   [1]
        #                   [2]]
        # batched_idx= [[0 5]
        #               [1 5]
        #               [2 5]]

        # how much the demand is satisfied
        d_sat = tf.gather_nd(self.demand, batched_idx) #假设访问到该点，顾客需求量就被全部满足
        # 初始时候batched_idx= [[0 5]
        #                      [1 5]
        #                      [2 5]]
        # tf.gather_nd(self.demand,batched_idx)取的是depot的demand，每一个batch均为0
        # tf.gather_nd 可以收集n-dimension的tensor

        # update the demand
        d_scatter = tf.scatter_nd(batched_idx, d_sat, tf.cast(tf.shape(self.demand), tf.int64))
        # scatter_nd(
        #     indices,
        #     updates,
        #     shape,
        #     name=None
        # )
        # 此函数将返回一个Tensor,它与updates有相同的类型；根据indices应用的一个新具有给定的形状和更新的张量.
        # 往一个初始值为0 形状为demand、类型与d_sat相同的tensor，在位置batch_idx上插入值d_sat

        # 更新demand的值（demand减去当前已经满足的量）
        self.demand = tf.subtract(self.demand, d_scatter)
        self.stage = stage
        # update load
        self.load -= d_sat #由于已经假设卡车为无限容量，此时不需要将卡车装满

        # 将demand为0的顾客（即访问过的）的mask置为1，depot的mask始终为1
        self.mask = tf.concat([tf.cast(tf.equal(self.demand, 0), tf.float32)[:, :-1],
                               tf.ones([self.batch_beam, 1])], 1)

        self.this_mask = tf.expand_dims(self.mask,0)

        # def first_stage():
        #     self.next_mask = self.total_mask[self.stage + 1:]
        #     return tf.concat([self.this_mask, self.next_mask], 0)
        #
        # def other_stage():
        #     self.pre_mask = self.total_mask[:self.stage]
        #     return tf.cond(tf.equal(self.stage,self.stage_num - 1), concat_first_two_parts, concat_three_parts)
        #
        # def concat_first_two_parts():
        #     return tf.concat([self.pre_mask, self.this_mask], 0)
        #
        # def concat_three_parts():
        #     self.next_mask = self.total_mask[self.stage + 1:]
        #     return tf.concat([self.pre_mask, self.this_mask, self.next_mask], 0)

        # self.total_mask = tf.cond(tf.cast(self.stage>0,tf.bool), other_stage, first_stage)

        state = State(load=self.load,
                      demand=self.demand,
                      d_sat=d_sat,
                      mask=self.mask,
                      # total_mask=self.total_mask,
                      start_time=self.start_time,
                      end_time=self.end_time)
        return state

# # 最小拖期
# def reward_func(state,env):
#     print("reward_Func")
#     superBigValue = 10000
#     total_mask = tf.stack(state[4], 0)
#     best_mask = tf.concat(
#         [tf.ones((env.stage_num, env.batch_size, env.n_nodes - 1)), tf.zeros((env.stage_num, env.batch_size, 1))],
#         axis=2)
#     mask_penalty = tf.reduce_sum(tf.reduce_sum((best_mask - total_mask), axis=2), axis=0) * superBigValue
#     ddl_penalty = tf.reduce_sum(tf.abs(tf.subtract(env.end_time[-1], env.ddl)), axis=1)
#     print("reward结束")
#     return tf.cast(mask_penalty + ddl_penalty, dtype=tf.float32)

#最小化makespan
def reward_func(state,env):
    print("reward_Func")
    superBigValue = 10000
    # total_mask = tf.stack(state[4], 0)
    mask = tf.stack(state[3], 0)
    best_mask = tf.ones((env.batch_size, env.n_nodes))
    mask_penalty = tf.reduce_sum(tf.reduce_sum((best_mask - mask), axis=1), axis=0) * superBigValue
    makespan_penalty = tf.reduce_max(env.end_time[-1],1)
    print("reward结束")
    return tf.cast(mask_penalty + makespan_penalty, dtype=tf.float32)

def reward_func_stage(sample_solution,stage):
    print("reward_stage开始")
    # python list to Tensor
    sample_solution = tf.stack(sample_solution, 0)
    # drop first demand column
    sample_solution = sample_solution[:, :, :, 2+stage] # 第0位存放demand 第1位存放ddl
    # sum by node_num
    sumed = tf.reduce_sum(sample_solution, axis=1)
    # varience by depot_num
    mean, var = tf.nn.moments(sumed, axes=[0])
    print("reward_stage结束")
    return  tf.cast(var, dtype=tf.float32)


