import tensorflow as tf
import numpy as np
import time
from shared.embeddings import LinearEmbedding
from shared.decode_step import RNNDecodeStep
import random


class RLAgent(object):

    def __init__(self,
                 args,
                 prt,
                 env,
                 dataGen,
                 reward_func,
                 reward_func_stage,
                 clAttentionActor,
                 clAttentionCritic,
                 is_train=True,
                 _scope=''):
        self.inference__ = '''
        This class builds the model and run testt and train.
        Inputs:
            args: arguments. See the description in config.py file.
            prt: print controller which writes logs to a file.
            env: an instance of the environment.
            dataGen: a data generator which generates data for test and training.
            reward_func: the function which is used for computing the reward. In the
                        case of TSP and VRP, it returns the tour length.
            clAttentionActor: Attention mechanism that is used in actor.
            clAttentionCritic: Attention mechanism that is used in critic.
            is_train: if true, the agent is used for training; else, it is used only
                        for inference.
        '''
        #self.epsilon = tf.placeholder(tf.float32)
        self.args = args
        self.prt = prt
        self.env = env
        self.dataGen = dataGen
        self.reward_func = reward_func

        #self.reward_func_stage = reward_func_stage
        self.clAttentionCritic = clAttentionCritic

        self.embedding = LinearEmbedding(args['embedding_dim'],
                                       _scope=_scope + 'Actor/')
        self.decodeStep = RNNDecodeStep(clAttentionActor,
                                        args['hidden_dim'],
                                        use_tanh=args['use_tanh'],
                                        tanh_exploration=args['tanh_exploration'],
                                        n_glimpses=args['n_glimpses'],
                                        mask_glimpses=args['mask_glimpses'],
                                        mask_pointer=args['mask_pointer'],
                                        forget_bias=args['forget_bias'],
                                        rnn_layers=args['rnn_layers'],
                                        _scope='Actor/')
        self.decoder_input = tf.get_variable('decoder_input', [1, 1, args['embedding_dim']],
                                             initializer=tf.contrib.layers.xavier_initializer())

        start_time = time.time()
        if is_train:
            self.train_summary = self.build_model(decode_type="stochastic")
            self.train_step = self.build_train_step()
        # for evaluation
        self.val_summary_greedy = self.build_model(decode_type="greedy")
        # self.val_summary_beam = self.build_model(decode_type = "beam_search")

        model_time = time.time() - start_time
        self.prt.print_out("It took {}s to build the agent.".format(str(model_time)))

        self.saver = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    # def build_model(self, decode_type="greedy"):
    #
    #     # builds the model
    #     args = self.args
    #     env = self.env
    #     batch_size = tf.shape(env.input_pnt)[0]
    #     process_time = env.process_time
    #     # input_pnt: [batch_size x max_time x input_dim]
    #
    #     # encoder_emb_inp: [batch_size, max_time, embedding_dim]
    #     # 这里我把输入改为只有加工时间
    #     encoder_emb_inp = self.embedding(process_time)
    #
    #     if decode_type == 'greedy' or decode_type == 'stochastic':
    #         beam_width = 1
    #     elif decode_type == 'beam_search':
    #         beam_width = args['beam_width']
    #
    #     # reset the env. The environment is modified to handle beam_search decoding.
    #     env.each_step_reset(beam_width)
    #
    #     BatchSequence = tf.expand_dims(tf.cast(tf.range(batch_size * beam_width), tf.int64), 1)
    #     # create tensors and lists
    #     actions_total = []
    #     idxs_total = []
    #     logprobs = []
    #     probs = []
    #     variance_total = tf.zeros((batch_size))
    #
    #     # start from depot
    #     idx = (env.n_nodes - 1) * tf.ones([batch_size * beam_width, 1])
    #     # tensorflow中的tile()函数是用来对张量(Tensor)进行扩展的，其特点是对当前张量内的数据进行一定规则的复制。
    #     action = tf.tile(process_time[:, env.n_nodes - 1], [beam_width, 1])
    #
    #     # decoder_state
    #     initial_state = tf.zeros([args['rnn_layers'], 2, batch_size * beam_width, args['hidden_dim']])
    #     l = tf.unstack(initial_state, axis=0)
    #     decoder_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
    #                            for idx in range(args['rnn_layers'])])
    #
    #     # start from depot in VRP and from a trainable nodes in TSP
    #     # decoder_input: [batch_size*beam_width x 1 x hidden_dim]
    #     if args['task_name'] == 'tsp':
    #         # decoder_input: [batch_size*beam_width x 1 x hidden_dim]
    #         decoder_input = tf.tile(self.decoder_input, [batch_size * beam_width, 1, 1])
    #     elif args['task_name'] == 'vrp':
    #         decoder_input = tf.tile(tf.expand_dims(encoder_emb_inp[:, env.n_nodes - 1], 1),
    #                                 [beam_width, 1, 1])
    #
    #     # decoding loop
    #
    #     context = tf.tile(encoder_emb_inp, [beam_width, 1, 1])
    #     for stage in range(args['stage_num']):
    #         env.each_stage_reset(stage)
    #         stage_idxs = []
    #         stage_actions = []
    #         for i in range(args['n_nodes']-1):
    #
    #             logit, prob, logprob, decoder_state = self.decodeStep.step(decoder_input,
    #                                                                        context,
    #                                                                        env,
    #                                                                        decoder_state)
    #
    #             # idx: [batch_size*beam_width x 1]
    #             beam_parent = None
    #
    #             if decode_type == 'greedy':
    #                 idx = tf.expand_dims(tf.argmax(prob, 1), 1)
    #
    #             elif decode_type == 'stochastic':
    #                 # select stochastic actions. idx has shape [batch_size x 1]
    #                 # tf.multinomial sometimes gives numerical errors, so we use our multinomial :(
    #                 def my_multinomial():
    #                     prob_idx = tf.stop_gradient(prob)
    #                     prob_idx_cum = tf.cumsum(prob_idx, 1)
    #                     rand_uni = tf.tile(tf.random_uniform([batch_size, 1]), [1, env.n_nodes])
    #                     # sorted_ind : [[0,1,2,3..],[0,1,2,3..] , ]
    #                     sorted_ind = tf.cast(tf.tile(tf.expand_dims(tf.range(env.n_nodes), 0), [batch_size, 1]),
    #                                          tf.int64)
    #                     tmp = tf.multiply(tf.cast(tf.greater(prob_idx_cum, rand_uni), tf.int64), sorted_ind) + \
    #                           10000 * tf.cast(tf.greater_equal(rand_uni, prob_idx_cum), tf.int64)
    #
    #                     idx = tf.expand_dims(tf.argmin(tmp, 1), 1)
    #                     return tmp, idx
    #
    #                 tmp, idx = my_multinomial()
    #                 # check validity of tmp -> True or False -- True mean take a new sample
    #                 tmp_check = tf.cast(
    #                     tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(tmp, 1), (10000 * env.n_nodes) - 1),
    #                                           tf.int32)), tf.bool)
    #                 tmp, idx = tf.cond(tmp_check, my_multinomial, lambda: (tmp, idx))
    #
    #             elif decode_type == 'beam_search':
    #                 if i == 0:
    #                     # BatchBeamSeq: [batch_size*beam_width x 1]
    #                     # [0,1,2,3,...,127,0,1,...],
    #                     batchBeamSeq = tf.expand_dims(tf.tile(tf.cast(tf.range(batch_size), tf.int64),
    #                                                           [beam_width]), 1)
    #                     beam_path = []
    #                     log_beam_probs = []
    #                     # in the initial decoder step, we want to choose beam_width different branches
    #                     # log_beam_prob: [batch_size, sourceL]
    #                     log_beam_prob = tf.log(tf.split(prob, num_or_size_splits=beam_width, axis=0)[0])
    #
    #                 elif i > 0:
    #                     log_beam_prob = tf.log(prob) + log_beam_probs[-1]
    #                     # log_beam_prob:[batch_size, beam_width*sourceL]
    #                     log_beam_prob = tf.concat(tf.split(log_beam_prob, num_or_size_splits=beam_width, axis=0), 1)
    #
    #                 # topk_prob_val,topk_logprob_ind: [batch_size, beam_width]
    #                 topk_logprob_val, topk_logprob_ind = tf.nn.top_k(log_beam_prob, beam_width)
    #
    #                 # topk_logprob_val , topk_logprob_ind: [batch_size*beam_width x 1]
    #                 topk_logprob_val = tf.transpose(tf.reshape(
    #                     tf.transpose(topk_logprob_val), [1, -1]))
    #
    #                 topk_logprob_ind = tf.transpose(tf.reshape(
    #                     tf.transpose(topk_logprob_ind), [1, -1]))
    #
    #                 # idx,beam_parent: [batch_size*beam_width x 1]
    #                 idx = tf.cast(topk_logprob_ind % env.n_nodes, tf.int64)  # Which city in route.
    #                 beam_parent = tf.cast(topk_logprob_ind // env.n_nodes, tf.int64)  # Which hypothesis it came from.
    #
    #                 # batchedBeamIdx:[batch_size*beam_width]
    #                 batchedBeamIdx = batchBeamSeq + tf.cast(batch_size, tf.int64) * beam_parent
    #                 prob = tf.gather_nd(prob, batchedBeamIdx)
    #
    #                 beam_path.append(beam_parent)
    #                 log_beam_probs.append(topk_logprob_val)
    #
    #             state = env.step(idx, stage, beam_parent)
    #
    #             batched_idx = tf.concat([BatchSequence, idx], 1)
    #
    #             decoder_input = tf.expand_dims(tf.gather_nd(
    #                 tf.tile(encoder_emb_inp, [beam_width, 1, 1]), batched_idx), 1)
    #
    #             logprob = tf.log(tf.gather_nd(prob, batched_idx))
    #             probs.append(prob)
    #             logprobs.append(logprob)
    #             action = tf.gather_nd(tf.tile(process_time, [beam_width, 1, 1]), batched_idx)
    #
    #             stage_actions.append(action)
    #             stage_idxs.append(idx)
    #
    #
    #
    #        # stage_action（node_num-1 * batch_size*input_dim）
    #         #R_stage = self.reward_func_stage(stage_actions,stage)
    #         stage_idxs = tf.squeeze(stage_idxs,2)  # （depot_num * node_num * batch_size*1）
    #
    #         state = env.update_start_end_time(stage_idxs,stage)
    #         actions_total.append(stage_actions)
    #         idxs_total.append(stage_idxs)
    #
    #
    #     R = self.reward_func(state,env)
    #
    #     ### critic
    #     v = tf.constant(0)
    #     if decode_type == 'stochastic':
    #         with tf.variable_scope("Critic"):
    #             with tf.variable_scope("Encoder"):
    #                 # init states
    #                 initial_state = tf.zeros([args['rnn_layers'], 2, batch_size, args['hidden_dim']])
    #                 l = tf.unstack(initial_state, axis=0)
    #                 rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
    #                                          for idx in range(args['rnn_layers'])])
    #
    #                 hy = rnn_tuple_state[0][1]
    #
    #             with tf.variable_scope("Process"):
    #                 for i in range(args['n_process_blocks']):
    #                     process = self.clAttentionCritic(args['hidden_dim'], _name="P" + str(i))
    #                     e, logit = process(hy, encoder_emb_inp, env)
    #
    #                     prob = tf.nn.softmax(logit)
    #                     # hy : [batch_size x 1 x sourceL] * [batch_size  x sourceL x hidden_dim]  ->
    #                     # [batch_size x h_dim ]
    #                     hy = tf.squeeze(tf.matmul(tf.expand_dims(prob, 1), e), 1)
    #
    #             with tf.variable_scope("Linear"):
    #                 v = tf.squeeze(tf.layers.dense(tf.layers.dense(hy, args['hidden_dim'] \
    #                                                                , tf.nn.relu, name='L1'), 1, name='L2'), 1)
    #
    #     return (R, variance_total, v, logprobs, actions_total, idxs_total, env.input_pnt, probs, state)

    def build_model(self, decode_type="greedy"):

        # builds the model
        args = self.args
        env = self.env
        batch_size = tf.shape(env.input_pnt)[0]
        process_time = env.process_time
        # input_pnt: [batch_size x max_time x input_dim]

        # encoder_emb_inp: [batch_size, max_time, embedding_dim]
        # 这里我把输入改为只有加工时间
        encoder_emb_inp = self.embedding(process_time)

        if decode_type == 'greedy' or decode_type == 'stochastic':
            beam_width = 1
        elif decode_type == 'beam_search':
            beam_width = args['beam_width']

        # reset the env. The environment is modified to handle beam_search decoding.
        env.each_step_reset(beam_width)

        BatchSequence = tf.expand_dims(tf.cast(tf.range(batch_size * beam_width), tf.int64), 1)
        # create tensors and lists
        actions_total = []
        idxs_total = []
        logprobs = []
        probs = []
        variance_total = tf.zeros((batch_size))

        # start from depot
        idx = (env.n_nodes - 1) * tf.ones([batch_size * beam_width, 1])
        # tensorflow中的tile()函数是用来对张量(Tensor)进行扩展的，其特点是对当前张量内的数据进行一定规则的复制。
        action = tf.tile(process_time[:, env.n_nodes - 1], [beam_width, 1])

        # decoder_state
        initial_state = tf.zeros([args['rnn_layers'], 2, batch_size * beam_width, args['hidden_dim']])
        l = tf.unstack(initial_state, axis=0)
        decoder_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
                               for idx in range(args['rnn_layers'])])

        # start from depot in VRP and from a trainable nodes in TSP
        # decoder_input: [batch_size*beam_width x 1 x hidden_dim]
        if args['task_name'] == 'tsp':
            # decoder_input: [batch_size*beam_width x 1 x hidden_dim]
            decoder_input = tf.tile(self.decoder_input, [batch_size * beam_width, 1, 1])
        elif args['task_name'] == 'vrp':
            decoder_input = tf.tile(tf.expand_dims(encoder_emb_inp[:, env.n_nodes - 1], 1),
                                    [beam_width, 1, 1])

        # decoding loop

        context = tf.tile(encoder_emb_inp, [beam_width, 1, 1])

        #env.each_stage_reset(stage)
        stage_idxs = []
        stage_actions = []
        for i in range(args['n_nodes']-1):

            logit, prob, logprob, decoder_state = self.decodeStep.step(decoder_input,
                                                                       context,
                                                                       env,
                                                                       decoder_state)

            # idx: [batch_size*beam_width x 1]
            beam_parent = None

            if decode_type == 'greedy':
                idx = tf.expand_dims(tf.argmax(prob, 1), 1)

            elif decode_type == 'stochastic':
                # select stochastic actions. idx has shape [batch_size x 1]
                # tf.multinomial sometimes gives numerical errors, so we use our multinomial :(
                def my_multinomial():
                    prob_idx = tf.stop_gradient(prob)
                    prob_idx_cum = tf.cumsum(prob_idx, 1)
                    rand_uni = tf.tile(tf.random_uniform([batch_size, 1]), [1, env.n_nodes])
                    # sorted_ind : [[0,1,2,3..],[0,1,2,3..] , ]
                    sorted_ind = tf.cast(tf.tile(tf.expand_dims(tf.range(env.n_nodes), 0), [batch_size, 1]),
                                         tf.int64)
                    tmp = tf.multiply(tf.cast(tf.greater(prob_idx_cum, rand_uni), tf.int64), sorted_ind) + \
                          10000 * tf.cast(tf.greater_equal(rand_uni, prob_idx_cum), tf.int64)

                    idx = tf.expand_dims(tf.argmin(tmp, 1), 1)
                    return tmp, idx

                tmp, idx = my_multinomial()
                # check validity of tmp -> True or False -- True mean take a new sample
                tmp_check = tf.cast(
                    tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(tmp, 1), (10000 * env.n_nodes) - 1),
                                          tf.int32)), tf.bool)
                tmp, idx = tf.cond(tmp_check, my_multinomial, lambda: (tmp, idx))

            elif decode_type == 'beam_search':
                if i == 0:
                    # BatchBeamSeq: [batch_size*beam_width x 1]
                    # [0,1,2,3,...,127,0,1,...],
                    batchBeamSeq = tf.expand_dims(tf.tile(tf.cast(tf.range(batch_size), tf.int64),
                                                          [beam_width]), 1)
                    beam_path = []
                    log_beam_probs = []
                    # in the initial decoder step, we want to choose beam_width different branches
                    # log_beam_prob: [batch_size, sourceL]
                    log_beam_prob = tf.log(tf.split(prob, num_or_size_splits=beam_width, axis=0)[0])

                elif i > 0:
                    log_beam_prob = tf.log(prob) + log_beam_probs[-1]
                    # log_beam_prob:[batch_size, beam_width*sourceL]
                    log_beam_prob = tf.concat(tf.split(log_beam_prob, num_or_size_splits=beam_width, axis=0), 1)

                # topk_prob_val,topk_logprob_ind: [batch_size, beam_width]
                topk_logprob_val, topk_logprob_ind = tf.nn.top_k(log_beam_prob, beam_width)

                # topk_logprob_val , topk_logprob_ind: [batch_size*beam_width x 1]
                topk_logprob_val = tf.transpose(tf.reshape(
                    tf.transpose(topk_logprob_val), [1, -1]))

                topk_logprob_ind = tf.transpose(tf.reshape(
                    tf.transpose(topk_logprob_ind), [1, -1]))

                # idx,beam_parent: [batch_size*beam_width x 1]
                idx = tf.cast(topk_logprob_ind % env.n_nodes, tf.int64)  # Which city in route.
                beam_parent = tf.cast(topk_logprob_ind // env.n_nodes, tf.int64)  # Which hypothesis it came from.

                # batchedBeamIdx:[batch_size*beam_width]
                batchedBeamIdx = batchBeamSeq + tf.cast(batch_size, tf.int64) * beam_parent
                prob = tf.gather_nd(prob, batchedBeamIdx)

                beam_path.append(beam_parent)
                log_beam_probs.append(topk_logprob_val)

            state = env.step(idx, 0, beam_parent)

            batched_idx = tf.concat([BatchSequence, idx], 1)

            decoder_input = tf.expand_dims(tf.gather_nd(
                tf.tile(encoder_emb_inp, [beam_width, 1, 1]), batched_idx), 1)

            logprob = tf.log(tf.gather_nd(prob, batched_idx))
            probs.append(prob)
            logprobs.append(logprob)
            action = tf.gather_nd(tf.tile(process_time, [beam_width, 1, 1]), batched_idx)

            stage_actions.append(action)
            stage_idxs.append(idx)

        stage_idxs = tf.squeeze(stage_idxs, 2)  # （depot_num * node_num * batch_size*1）
        for stage in range(args['stage_num']):
            state = env.update_start_end_time(stage_idxs,stage)
            actions_total.append(stage_actions)
            idxs_total.append(stage_idxs)


        R = self.reward_func(state,env)

        ### critic
        v = tf.constant(0)
        if decode_type == 'stochastic':
            with tf.variable_scope("Critic"):
                with tf.variable_scope("Encoder"):
                    # init states
                    initial_state = tf.zeros([args['rnn_layers'], 2, batch_size, args['hidden_dim']])
                    l = tf.unstack(initial_state, axis=0)
                    rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
                                             for idx in range(args['rnn_layers'])])

                    hy = rnn_tuple_state[0][1]

                with tf.variable_scope("Process"):
                    for i in range(args['n_process_blocks']):
                        process = self.clAttentionCritic(args['hidden_dim'], _name="P" + str(i))
                        e, logit = process(hy, encoder_emb_inp, env)

                        prob = tf.nn.softmax(logit)
                        # hy : [batch_size x 1 x sourceL] * [batch_size  x sourceL x hidden_dim]  ->
                        # [batch_size x h_dim ]
                        hy = tf.squeeze(tf.matmul(tf.expand_dims(prob, 1), e), 1)

                with tf.variable_scope("Linear"):
                    v = tf.squeeze(tf.layers.dense(tf.layers.dense(hy, args['hidden_dim'] \
                                                                   , tf.nn.relu, name='L1'), 1, name='L2'), 1)

        return (R, variance_total, v, logprobs, actions_total, idxs_total, env.input_pnt, probs, state)

    def build_train_step(self):
        '''
        This function returns a train_step op, in which by running it we proceed one training step.
        '''
        args = self.args

        R, variance_total, v, logprobs, actions, idxs, batch, probs, state = self.train_summary

        v_nograd = tf.stop_gradient(v)
        R = tf.stop_gradient(R)

        # losses
        actor_loss = tf.reduce_mean(tf.multiply((R - v_nograd), tf.add_n(logprobs)), 0)
        critic_loss = tf.losses.mean_squared_error(R, v)

        # optimizers
        actor_optim = tf.train.AdamOptimizer(args['actor_net_lr'])
        critic_optim = tf.train.AdamOptimizer(args['critic_net_lr'])

        # compute gradients
        actor_gra_and_var = actor_optim.compute_gradients(actor_loss, \
                                                          tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                            scope='Actor'))
        critic_gra_and_var = critic_optim.compute_gradients(critic_loss, \
                                                            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                              scope='Critic'))

        # clip gradients
        clip_actor_gra_and_var = [(tf.clip_by_norm(grad, args['max_grad_norm']), var) \
                                  for grad, var in actor_gra_and_var]

        clip_critic_gra_and_var = [(tf.clip_by_norm(grad, args['max_grad_norm']), var) \
                                   for grad, var in critic_gra_and_var]

        # apply gradients
        actor_train_step = actor_optim.apply_gradients(clip_actor_gra_and_var)
        critic_train_step = critic_optim.apply_gradients(clip_critic_gra_and_var)

        train_step = [actor_train_step,
                      critic_train_step,
                      actor_loss,
                      critic_loss,
                      actor_gra_and_var,
                      critic_gra_and_var,
                      R,
                      variance_total,
                      v,
                      logprobs,
                      probs,
                      actions,
                      idxs]
        return train_step

    def Initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.load_model()
        print("finish attention_agent initialize!")

    def load_model(self):
        print("load_model!")
        latest_ckpt = tf.train.latest_checkpoint(self.args['load_path'])
        if latest_ckpt is not None:
            print("checkpoint not null!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.saver.restore(self.sess, latest_ckpt)

    def evaluate_single(self, eval_type='greedy'):
        print("enter evaluate_single!")
        start_time = time.time()
        avg_reward = []

        if eval_type == 'greedy':
            summary = self.val_summary_greedy
        elif eval_type == 'beam_search':
            summary = self.val_summary_beam
        self.dataGen.reset()
        for step in range(self.dataGen.n_problems):

            data = self.dataGen.get_test_next()

            R, variance_total, v, logprobs, actions, idxs, batch, _, state = self.sess.run(summary,
                                                                           feed_dict={self.env.input_data: data,
                                                                                      self.decodeStep.dropout: 0.0,
                                                                                      # self.epsilon: self.args[
                                                                                      #     'epsilon']
                                                                                      })
            if eval_type == 'greedy':
                avg_reward.append(R)
                R_ind0 = 0
            elif eval_type == 'beam_search':
                # R : [batch_size x beam_width]
                R = np.concatenate(np.split(np.expand_dims(R, 1), self.args['beam_width'], axis=0), 1)
                R_val = np.amin(R, 1, keepdims=False)
                R_ind0 = np.argmin(R, 1)[0]
                avg_reward.append(R_val)

            # sample decode

            example_output = []
            example_input = []
            for i in range(self.env.n_nodes):
                example_input.append(list(batch[0, i, :]))
            for id,stage_action in enumerate(actions):
                example_output.append(np.squeeze(stage_action))

            idx = []
            for i in range(len(idxs)):
                temp = np.squeeze(idxs[i], 1)
                idx.append(temp.tolist())

            self.prt.print_out('\n\nVal-Step of {}: {}'.format(eval_type, step))
            self.prt.print_out('\nExample test input: {}'.format(example_input))
            #self.prt.print_out('\nExample test output: {}'.format(example_output))
            self.prt.print_out('\nExample test reward: {} - best: {}'.format(R[0], R_ind0))
            self.prt.print_out('\ntotal_idx: {}'.format(idx))
            self.prt.print_out('\nvariance_total: {}'.format(variance_total[0]))



        end_time = time.time() - start_time

        # Finished going through the iterator dataset.
        self.prt.print_out('\nValidation overall avg_reward: {}'.format(np.mean(avg_reward)))
        self.prt.print_out('Validation overall reward std: {}'.format(np.sqrt(np.var(avg_reward))))

        self.prt.print_out("Finished evaluation with %d steps in %s." % (step \
                                                                             , time.strftime("%H:%M:%S",
                                                                                             time.gmtime(end_time))))

    def evaluate_batch(self, eval_type='greedy'):
        print("enter evaluate_batch!")
        self.env.each_step_reset()
        if eval_type == 'greedy':
            summary = self.val_summary_greedy
            beam_width = 1
        elif eval_type == 'beam_search':
            summary = self.val_summary_beam
            beam_width = self.args['beam_width']

        data = self.dataGen.get_test_all()
        start_time = time.time()
        R, variance_total, v, logprobs, actions, idxs, batch, _, _ = self.sess.run(summary,
                                                                   feed_dict={self.env.input_data: data,
                                                                              self.decodeStep.dropout: 0.0,
                                                                              # self.epsilon: self.args['epsilon'],
                                                                              })
        R = np.concatenate(np.split(np.expand_dims(R, 1), beam_width, axis=0), 1)
        R = np.amin(R, 1, keepdims=False)

        end_time = time.time() - start_time
        self.prt.print_out('Average of {} in batch-mode: {} -- std {} -- time {} s'.format(eval_type, \
                                                                                           np.mean(R),
                                                                                           np.sqrt(np.var(R)),
                                                                                           end_time))

    def inference(self, infer_type='batch'):
        if infer_type == 'batch':
            self.evaluate_batch('greedy')
            # self.evaluate_batch('beam_search')
        elif infer_type == 'single':
            self.evaluate_single('greedy')
            # self.evaluate_single('beam_search')
        self.prt.print_out("##################################################################")

    def run_train_step(self):
        data = self.dataGen.get_train_next()
        # self.args['epsilon'] *= 0.999

        train_results = self.sess.run(self.train_step,
                                      feed_dict={self.env.input_data: data,
                                                 self.decodeStep.dropout: self.args['dropout'],
                                                 # self.epsilon: self.args['epsilon']
                                                 })
        return train_results
