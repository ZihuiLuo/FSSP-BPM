import numpy as np
import tensorflow as tf
import os
import warnings
import collections
import pandas as pd


def load_test_dataset(args):
    n_problems, n_cust, data_dir = args['test_size'], args['n_cust'], args['data_dir']
    input_dim = args['input_dim']
    print('load test data.')
    fname = os.path.join(data_dir, "test.txt")

    # load test data
    if os.path.exists(fname):
        data = np.loadtxt(fname)

        if len(data) < n_cust:
            append_len = n_cust - len(data)
            zeros = np.zeros((append_len, input_dim))
            data = np.concatenate([data, zeros])
            np.savetxt(fname, data, fmt='%f', delimiter='\t')

        row_rand_array = np.arange(data.shape[0])
        np.random.shuffle(row_rand_array)
        data = data[row_rand_array[0:(n_problems * n_cust)]].reshape(n_problems, n_cust, input_dim)
        depot = np.zeros((n_problems, 1, input_dim))
        input_data = np.concatenate([data, depot], 1)
    else:
        print('no test data!')
    return input_data


class DataGenerator(object):
    def __init__(self, args):

        self.args = args
        # create test data
        self.n_problems = args['test_size']
        self.input_dim = args['input_dim']
        self.test_data = load_test_dataset(args)
        self.train_data = np.loadtxt(os.path.join(self.args['data_dir'], "train.txt"))
        self.count = 0
        self.reset()

    def reset(self):
        self.shuffle_data()

    def shuffle_data(self):
        # shuffle_data在每个epoch开始都执行
        self.count = 0
        row_rand_array = np.arange(self.train_data.shape[0])
        np.random.shuffle(row_rand_array)
        self.train_data = self.train_data[row_rand_array]

    def get_train_next(self):
        '''
        Get next batch of problems for training
        Retuens:
            input_data: data with shape [batch_size x max_time x input_dim]
        '''
        # load train data
        train_batch = self.train_data[self.count:(self.count + self.args['batch_size'] * self.args['n_cust'])]
        self.count = self.count + self.args['batch_size'] * self.args['n_cust']

        cust_data = np.array(train_batch).reshape(self.args['batch_size'], self.args['n_cust'], self.input_dim)
        depot = np.zeros((self.args['batch_size'], 1, self.input_dim))
        batch_data = np.concatenate([cust_data, depot], 1)

        return batch_data

    def get_test_next(self):
        '''
        Get next batch of problems for testing
        '''
#        if self.count < self.args['test_size']:
#            input_pnt = self.test_data[self.count:self.count + 1]
#            self.count += 1
#        else:
#            warnings.warn("The test iterator reset.")
#            self.count = 0
#            input_pnt = self.test_data[self.count:self.count + 1]
#            self.count += 1

        return self.test_data

    def get_test_all(self):
        '''
        Get all test problems
        '''
        return self.test_data


class State(collections.namedtuple("State",
                                   ("load",
                                    "demand",
                                    'd_sat',
                                    "mask"))):
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
        self.input_data = tf.placeholder(tf.float32, \
                                         shape=[None, self.n_nodes, self.input_dim])
        # 各个数据的id
        self.data_id = self.input_data[:, :, 0]
        # self.input_pnt表示数据各维数据
        self.input_pnt = self.input_data[:, :, 1:self.input_dim-1]
        self.demand = self.input_data[:, :, -1]
        self.batch_size = tf.shape(self.input_pnt)[0]

    def reset(self, beam_width=1):
        '''
        Resets the environment. This environment might be used with different decoders.
        In case of using with beam-search decoder, we need to have to increase
        the rows of the mask by a factor of beam_width.
        '''

        # dimensions
        self.beam_width = beam_width
        self.batch_beam = self.batch_size * beam_width
        self.data_id = self.input_data[:, :, 0]
        self.input_pnt = self.input_data[:, :, 1:self.input_dim-1]
        self.demand = self.input_data[:, :, -1]


        # modify the self.input_pnt and self.demand for beam search decoder
        self.input_pnt = tf.tile(self.input_pnt, [self.beam_width,1,1])

        # demand: [batch_size * beam_width, max_time]
        # demand[i] = demand[i+batchsize]
        self.demand = tf.tile(self.demand, [self.beam_width, 1])

        # load: [batch_size * beam_width]
        self.load = tf.ones([self.batch_beam]) * self.capacity

        # create mask
        self.mask = tf.zeros([self.batch_size * beam_width, self.n_nodes],
                             dtype=tf.float32)

        # update mask
        # 改成不允许split delivery 即增加条件：当顾客的demand大于当前汽车的load,则将顾客的mask置为1
        # 当顾客demand==0，也将顾客mask置为1； depot的mask初始化都为1，表示不可访问
        # self.mask = tf.concat([tf.cast(tf.equal(self.demand, 0), tf.float32)[:, :-1],
        #                        tf.ones([self.batch_beam, 1])], 1)
        # self.mask = tf.concat([tf.multiply(tf.cast(tf.equal(self.demand, 0), tf.float32),
        #                                    tf.cast(tf.greater(self.demand, np.tile(self.load,self.n_nodes)), tf.float32))[:, :-1],
        #                        tf.ones([self.batch_beam, 1])], 1)
        self.mask = tf.concat([
            tf.cast(
                tf.logical_or(tf.equal(self.demand, 0),
                              tf.greater(self.demand, tf.tile(tf.expand_dims(self.load,1),[1,self.n_nodes]))
                              ), tf.float32)[:, :-1], tf.ones([self.batch_beam, 1])], 1)

        # mask的形状为batch_beam * n_nodes

        state = State(load=self.load,
                      demand=self.demand,
                      d_sat=tf.zeros([self.batch_beam, self.n_nodes]),
                      mask=self.mask)

        return state

    def step(self,
             idx,
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
        # d_sat = tf.minimum(tf.gather_nd(self.demand, batched_idx), self.load)
        d_sat = tf.gather_nd(self.demand, batched_idx) # 不允许split delivery 情况下,d_sat==顾客的demand

        # 将当前车的剩余负载与顾客的需求进行比较，取小的那个值作为该顾客的已满足量
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

        # update load
        self.load -= d_sat

        # refill the truck -- idx: [10,9,10] -> load_flag: [1 0 1]
        load_flag = tf.squeeze(tf.cast(tf.equal(idx, self.n_cust), tf.float32), 1)
        # tf.squeeze 该函数返回一个张量，这个张量是将原始input中所有维度为1的那些维都删掉的结果
        # 若batch_size为3，初始卡车从depot出发，此时load_flag为[1. 1. 1.]
        # 表示若此时回到depot则load_flag置为1，否则为0

        self.load = tf.multiply(self.load, 1 - load_flag) + load_flag * self.capacity
        # 若load_flag为1，表示此时卡车回到depot,则将卡车卡车当前容量装满，即将load置为capacity,否则依旧为load值

        # 将此时demand为0的顾客或者demand大于load的顾客的mask置为1
        # self.mask = tf.concat([tf.multiply(tf.cast(tf.equal(self.demand, 0), tf.float32),
        #                                    tf.cast(tf.greater(self.demand, np.tile(self.load, self.n_nodes)),
        #                                            tf.float32))[:, :-1],
        #                        tf.ones([self.batch_beam, 1])], 1)
        self.mask = tf.concat([
            tf.cast(
                tf.logical_or(tf.equal(self.demand, 0),
                              tf.greater(self.demand, tf.tile(tf.expand_dims(self.load, 1), [1, self.n_nodes]))
                              ), tf.float32)[:, :-1], tf.zeros([self.batch_beam, 1])], 1)

        # mask if load= 0
        # mask if in depot and there is still a demand

        # self.mask += tf.concat([tf.tile(tf.expand_dims(tf.cast(tf.equal(self.load, 0),
        #                                                        tf.float32), 1), [1, self.n_cust]),
        #                         tf.expand_dims(
        #                             tf.multiply(tf.cast(tf.greater(tf.reduce_sum(self.demand, 1), 0), tf.float32),
        #                                         tf.squeeze(tf.cast(tf.equal(idx, self.n_cust), tf.float32))), 1)], 1)

        self.mask += tf.concat([tf.tile(tf.expand_dims(tf.cast(tf.equal(self.load, 0),
                                                               tf.float32), 1), [1, self.n_cust]),
                                tf.expand_dims(
                                    tf.multiply(tf.cast(tf.greater(tf.reduce_sum(self.demand, 1), 0), tf.float32),
                                                tf.squeeze(tf.cast(tf.equal(idx, self.n_cust), tf.float32))), 1)], 1)
        # tf.tile(tf.expand_dims(tf.cast(tf.equal(self.load,0), tf.float32),1), [1,self.n_cust]
        # 表达式1表示若卡车的load为0，则该路径上所有节点值均为1,否则全为0
        # tf.cast(tf.greater(tf.reduce_sum(self.demand,1),0),tf.float32) 表示batch中如果有demand大于0
        # tf.squeeze( tf.cast(tf.equal(idx,self.n_cust),tf.float32)) 表示若卡车在depot
        # 表达式2表示若该batch中卡车在depot且路线中还有顾客的demand大于0，则对应depot的mask为1，即卡车下一步不能访问depot
        # 将表达式1和表达式2 concate起来

        state = State(load=self.load,
                      demand=self.demand,
                      d_sat=d_sat,
                      mask=self.mask)

        return state


def reward_func(sample_solution):
    """The reward for the VRP task is defined as the
    negative value of the route length

    Args:
        sample_solution : a list tensor of size decode_len of shape [batch_size x input_dim]
        demands satisfied: a list tensor of size decode_len of shape [batch_size]

    Returns:
        rewards: tensor of size [batch_size]

    Example:
        sample_solution = [[[1,1]
                          [2,2]],
                        [[3,3]
                        [4,4]],
                        [[5,5]
                        [6,6]]]
        sourceL = 3
        batch_size = 2
        input_dim = 2
        sample_solution_tilted[ [[5,5]
                                [6,6]],
                                [[1,1]
                                [2,2]],
                                [[3,3]
                                [4,4]] ]
    """
    # sample_solution 要求只包含A,B,C,D,E,F,H,I,art,thickness,width,date,priority,length 十四个属性
    # 将sample_solution由python list转为tensor, sample_solution.shape=(sourceL,batch_size,input_dim)
    sample_solution = tf.stack(sample_solution, 0)
    # 将sample_solution后推一个位置, shape=(sourceL,batch_size,input_dim)
    sample_solution_tilted = tf.concat((tf.expand_dims(sample_solution[-1], 0), sample_solution[:-1]), 0)
    # dis_list: 所有路径各段距离（包括配送中心到顾客的）, shape=(sourceL, batch_size)

    # class(A,B,C,D,E,F,H,I),art,thickness,width,date,priority,length属性权重
    score = tf.constant([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2, 0.2,0.2,0.16,0.1,0.1,0.04])
    #dis_list = tf.pow(tf.reduce_sum(tf.pow((sample_solution_tilted - sample_solution), 2), 2), .5)
    dis_list = tf.pow(tf.reduce_sum(tf.multiply(tf.pow((sample_solution_tilted - sample_solution), 2), score), 2), .5)

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
    route_lens_decoded = tf.reduce_sum(tf.multiply(dis_list, mask), axis=0) #从dis_list中扣除与配送中心相连的距离

    # non_zeros_pos为dis_list中路径长度不为0的段（所有顾客选择完后，最后的路径都为配送中心，距离在dis_list中都为0，要把对这段的惩罚扣除，因为中途返回配送中心有惩罚）
    non_zeros_pos = tf.cast(tf.count_nonzero(tf.expand_dims(dis_list, axis=-1), axis=-1), dtype=tf.float32)
    # mask_flip为dis_list中返回配送中心的次数的两倍，这里就当返回配送中心总次数 惩罚的权重为2
    # mask_flip = tf.multiply(non_zeros_pos, (1 - mask) * 2)
    mask_flip = tf.multiply(non_zeros_pos, (1 - mask)*2)
    route_lens_center = tf.reduce_sum(mask_flip, axis=0) # 返回配送中心次数的总惩罚
    route_lens_decoded += route_lens_center # 订单之间距离差异值惩罚+返回配送中心次数的总惩罚 作为reward
    ##### reward = args['alpha']*route_lens_decoded + args['beta']*state[0]
    # state[0]是load，即当前卡车剩余容量，形状和route_lens_decoded一样

    return route_lens_decoded
