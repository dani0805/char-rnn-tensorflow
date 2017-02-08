import tensorflow as tf
from tensorflow.python.framework.tensor_shape import Dimension
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import numpy as np
from tensorflow.python.ops.math_ops import matmul
from tensorflow.python.ops.nn_ops import conv1d
from tensorflow.python.ops.rnn import dynamic_rnn, rnn
from tensorflow.python.ops.seq2seq import attention_decoder

from utils import TextLoader


class Model():
    def __init__(self, args, infer=False):
        # args.conv_layers: number of convolutional blocks 3xd + 3xd stride 2, it also defines the number of equivalent
        #     upsampling blocks in the decoder part 3xd + 3xd stride 1/2
        # args.embed_size: size of the character embeddings
        # args.conv_size: number of channels in the first convolutional layer

        self.args = args
        if infer:
            args.batch_size = 1
            #args.seq_length = 128
        print("seq length: ", args.seq_length)
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.embedding = tf.get_variable("embedding", [args.ivocab_size, args.embed_size])
        self.baseconvfilters = tf.get_variable("baseconvfilters", [3, args.embed_size, args.conv_size],
            initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))

        conv_inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)
        base_outputs = tf.nn.relu(tf.nn.conv1d(conv_inputs, self.baseconvfilters, 1, "SAME"))
        rnn_inputs = self.multi_layer_conv(base_outputs, args)
        rnn_size = args.conv_size * 2 ** args.conv_layers

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))


        cell = cell_fn(rnn_size, state_is_tuple=True)
        cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=args.dropout)

        cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=True)
        self.cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=args.dropout)

        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        print(rnn_inputs.get_shape())
        rnn_inputs = tf.split(1,args.seq_length / 2 ** args.conv_layers,rnn_inputs)
        print(rnn_inputs[0].get_shape())
        rnn_inputs = list(map(tf.squeeze,rnn_inputs))
        print(rnn_inputs[0].get_shape())

        encoder_outputs, _ = rnn(
            self.cell,
            rnn_inputs,
            initial_state=self.initial_state
        )
        #encoder_outputs = [tf.reshape(x,(args.batch_size, 1, args.conv_size * 2 ** args.conv_layers)) for x in encoder_outputs]
        #encoder_outputs = tf.concat(1, encoder_outputs)
        #print(encoder_outputs.get_shape())
        self.attention_states = tf.get_variable("attentionstates", [args.batch_size, args.seq_length / 2 ** args.conv_layers, 4],
            initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))

        decoder_outputs, _ = attention_decoder(encoder_outputs, self.initial_state, self.attention_states, self.cell, num_heads=1)

        print(len(decoder_outputs), " x ", decoder_outputs[0].get_shape())
        decoder_outputs = [tf.reshape(x,(args.batch_size, 1, args.conv_size * 2 ** args.conv_layers)) for x in decoder_outputs]
        print(len(decoder_outputs), " x ", decoder_outputs[0].get_shape())
        decoder_outputs = tf.concat(1, decoder_outputs)
        print(decoder_outputs.get_shape())

        tconv_outputs = self.multi_layer_tconv(decoder_outputs,args)
        print(tconv_outputs.get_shape())
        tconv_outputs = tf.reshape( tconv_outputs, (-1, args.conv_size))
        softmax_w = tf.get_variable("softmax_w", [args.conv_size, args.tvocab_size])
        softmax_b = tf.get_variable("softmax_b", [args.tvocab_size])

        self.logits = tf.matmul(tconv_outputs, softmax_w) + softmax_b

        #self.logits =tf.reshape(self.logits, [-1, 256])
        print("logits: ",self.logits.get_shape())
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([args.batch_size * args.seq_length])],
            args.tvocab_size
        )
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def multi_layer_conv(self, inputs, args):
        self.conv_layer_filters = []
        out = inputs
        for layer in range(args.conv_layers):
            self.conv_layer_filters.append([
                tf.get_variable(
                    "conv_layer_{}_0".format(layer),
                    [3, args.conv_size * 2 ** layer, args.conv_size * 2 ** layer],
                    initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
                ),
                tf.get_variable(
                    "conv_layer_{}_1".format(layer),
                    [3, args.conv_size * 2 ** layer, args.conv_size * 2 ** layer * 2],
                    initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
                )
            ])
            out = tf.nn.relu(tf.nn.conv1d(out, self.conv_layer_filters[layer][0], 1, "SAME"))
            out = tf.nn.relu(tf.nn.conv1d(out, self.conv_layer_filters[layer][1], 2, "SAME"))
        return out

    def multi_layer_tconv(self, inputs, args):
        self.tconv_layer_filters = []
        out = inputs
        inputs_reshaped = array_ops.expand_dims(inputs, 1)
        for layer in range(args.conv_layers):
            self.tconv_layer_filters.append([
                tf.get_variable(
                    "tconv_layer_{}_0".format(layer),
                    [3, args.conv_size * 2 ** (args.conv_layers - layer), args.conv_size * 2 ** (args.conv_layers - layer)],
                    initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
                ),
                tf.get_variable(
                    "tconv_layer_{}_1".format(layer),
                    [3, args.conv_size * 2 ** (args.conv_layers - layer) / 2, args.conv_size * 2 ** (args.conv_layers - layer)],
                    initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
                )
            ])
            print(self.tconv_layer_filters[layer][0].get_shape())
            out = tf.nn.relu(tf.nn.conv1d(out, self.tconv_layer_filters[layer][0], 1, "SAME"))
            filters = array_ops.expand_dims(self.tconv_layer_filters[layer][1],0)
            out = tf.reshape(out,(
                args.batch_size,
                1,
                int(args.seq_length / 2 ** (args.conv_layers - layer)),
                int(args.conv_size * 2 ** (args.conv_layers - layer))
            ))
            print(out.get_shape())
            print(filters.get_shape())
            in_shape = out.get_shape()
            out_shape = [int(s) for s in in_shape]  # copy
            out_shape[2] = out_shape[2] * 2  # always true
            out_shape[3] = int(out_shape[3] / 2)  # always true
            print(out_shape)

            out = tf.nn.conv2d_transpose(
                out,
                filters,
                out_shape,
                [1,1,2,1],
                "SAME")
            out = tf.nn.relu(array_ops.squeeze(out,[1]))
        return out

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



def get_output_shape(in_layer, n_kernel, kernel_size, border_mode='same'):
	"""
	Always assumes stride=1
	"""
	in_shape = in_layer.get_shape() # assumes in_shape[0] = None or batch_size
	out_shape = [s for s in in_shape] # copy
	out_shape[-1] = n_kernel # always true
	if border_mode=='same':
		out_shape[1] = in_shape[1]
		out_shape[2] = in_shape[2]
	elif border_mode == 'valid':
		out_shape[1] = in_shape[1]+kernel_size - 1
		out_shape[2] = in_shape[2]+kernel_size - 1
	return out_shape