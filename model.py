import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import numpy as np
from tensorflow.python.ops.math_ops import matmul
from tensorflow.python.ops.nn_ops import conv1d

from utils import TextLoader


class Model():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            #args.seq_length = 128

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size, state_is_tuple=True)

        cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=args.dropout)

        self.output_size = np.trunc(args.seq_length / 2).astype(int)
        print("args.seq_length: ", args.seq_length)
        print("output_size: ",self.output_size)

        cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=True)
        self.cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=args.dropout)
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)


        softmax_w = tf.get_variable("softmax_w", [args.rnn_size, 2 * args.vocab_size])
        softmax_b = tf.get_variable("softmax_b", [2 * args.vocab_size])
        self.l1conv3filters = tf.get_variable("l1_conv_filter",[3, args.embed_size, args.rnn_size / 2],
            initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
        self.l2conv3filters = tf.get_variable("l2_conv_filter", [3, args.rnn_size / 2, args.rnn_size],
            initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))

        self.outputconv1filters = tf.get_variable("output_conv_filter", [1, args.rnn_size, args.embed_size],
            initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
        self.embedding = tf.get_variable("embedding", [args.vocab_size, args.embed_size])

        #inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(self.embedding, self.input_data))
        #inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        l1inputs = tf.nn.embedding_lookup(self.embedding, self.input_data )
        l1conv3 = tf.nn.relu(tf.nn.conv1d(l1inputs, self.l1conv3filters,1,"SAME"))
        l2conv3 = tf.nn.relu(tf.nn.conv1d(l1conv3, self.l2conv3filters, 1, "SAME"))
        print("l2conv3: ",l2conv3.get_shape())
        l2conv3_reshaped = tf.reshape(l2conv3,[args.batch_size, args.seq_length, 1, args.rnn_size ])
        print("l2conv3_reshaped: ",l2conv3_reshaped.get_shape())
        max_pool = tf.nn.max_pool(l2conv3_reshaped,ksize=[1, 3 , 1, 1],
            strides=[1, 2, 1, 1],
            padding='SAME',
            name="pool")
        print("max_pool: ", max_pool.get_shape())
        inputs = tf.split(1, self.output_size, tf.reshape(max_pool,[args.batch_size, self.output_size, args.rnn_size ]))
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            print("prev: ", prev.get_shape())
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(self.embedding, prev_symbol)

        print("inputs: ", len(inputs),"x",inputs[0].get_shape())
        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell,
            #loop_function=loop if infer else None,
            scope='rnnlm')
        tensoroutput = tf.reshape(tf.concat(1, outputs), [args.batch_size, self.output_size, args.rnn_size])
        #decoded_outputs = tf.nn.relu(tf.nn.conv1d(tensoroutput, self.outputconv1filters,1,"SAME"))
        output = tf.reshape(tensoroutput, [-1, args.rnn_size])
        print("output: ",output.get_shape())
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.logits =tf.reshape(self.logits, [-1, args.vocab_size])
        print("logits: ",self.logits.get_shape())
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, chars, vocab, data_dir, seq_length, num=200, prime='The ', sampling_type=1 ):
        data_loader = TextLoader(data_dir, 1, seq_length)
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return int(np.searchsorted(t, np.random.rand(1) * s))

        ret = prime
        char = prime[-1]
        _, y = data_loader.random_batch()
        #for i in range(128):
            #y[0, i] = vocab[char]
        #print("y.dtype: ", y.dtype)
        for n in range(num):

            feed = {self.input_data: y, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            #if n == 3:
            #print("probs :", probs[:2])
            pred = []
            sample = []
            for p in probs[:2]:
                if sampling_type == 0:
                    sample += [np.argmax(p)]
                    pred += [chars[sample[-1]]]
                elif sampling_type == 2:
                    if char == ' ':
                        sample += [weighted_pick(p)]
                        pred += [chars[sample[-1]]]
                    else:
                        sample += [np.argmax(p)]
                        pred += [chars[sample[-1]]]
                else: # sampling_type == 1 default:
                    sample += [weighted_pick(p)]
                    pred += [chars[sample[-1]]]
                char = pred[-1]

            #print("pred: ", pred)
            ret += "".join(pred)
            #print ("y.dtype: ", y.dtype)
            np.delete(y,np.s_[:2],1)
            #print("y.dtype: ", y.dtype)
            #print("ret: ", ret)

            #print("sample: ", sample)
            np.append(y, [sample], 1)
        return ret



