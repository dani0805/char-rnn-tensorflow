import codecs
import os
import collections
import numpy as np
import pickle


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, source_lang="en", target_lang="it", encoding='utf-8'):
        self.counter = 0
        self.bucket_pointer = 0
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "corpus.{}.txt".format(source_lang))
        target_file = os.path.join(data_dir, "corpus.{}.txt".format(target_lang))
        ivocab_file = os.path.join(data_dir, "inputvocab.pkl")
        tvocab_file = os.path.join(data_dir, "targetvocab.pkl")
        itensor_file = os.path.join(data_dir, "inputdata.npy")
        ttensor_file = os.path.join(data_dir, "targetdata.npy")

        if not (os.path.exists(ivocab_file) and os.path.exists(itensor_file) and os.path.exists(tvocab_file) and os.path.exists(ttensor_file)):
            print("reading text file")
            self.input_tensor = self.preprocess(input_file, ivocab_file, itensor_file)
            self.ichars, self.ivocab_size, self.ivocab, self.input_tensor, self.itensor_file = \
                self.preprocess(input_file, ivocab_file, itensor_file)
            self.tchars, self.tvocab_size, self.tvocab, self.target_tensor, self.ttensor_file = \
                self.preprocess(target_file, tvocab_file, ttensor_file)

        else:
            pass
            #print("loading preprocessed files")
            #self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as g:
            data = g.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        chars, _ = zip(*count_pairs)
        chars = ['_GO', '_END'] + list(chars)
        print(chars)
        vocab_size = len(chars)
        vocab = dict(zip(chars, range(len(chars))))
        with open(vocab_file, 'wb') as f:
            pickle.dump(chars, f)
        lines = []
        with codecs.open(input_file, "r", encoding=self.encoding) as g:
            for line in g:
                lines.append(list(map(vocab.get,line)))
        #print(lines)
        return chars, vocab_size, vocab, lines, tensor_file

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = pickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self, num_buckets=4):
        print(len(self.input_tensor), " " , len(self.input_tensor[0]), " ", self.input_tensor[0])
        tensor = np.array([self.input_tensor, self.target_tensor, [len(x) for x in self.input_tensor], [len(y) for y in self.target_tensor]])

        tensor = np.transpose(tensor, [1, 0])
        tensor = np.array(list(map(lambda x: [np.array(x[0]), np.array(x[1]), np.array([max(x[2], x[3])])], tensor)))

        tensor = tensor[tensor[:,2].argsort()]
        self.num_batches = int(tensor.size / self.batch_size)

        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.bucket_size = int(self.num_batches / num_buckets)
        self.buckets = []
        self.bucket_length = []

        for i in range(num_buckets):
            bucket = tensor[i*self.bucket_size:(i+1)*self.bucket_size]
            bucket_length = bucket[-1, 2][0] + 2
            bucket = np.delete(bucket, np.s_[2], axis=1)
            #print(bucket_length)

            #print(np.pad(bucket[0, 0],(0,bucket_length - bucket[0, 0].shape[0]),'constant',constant_values=(0,0)))
            #print(bucket[0, 0])
            #print(bucket[0, 1])
            #print(bucket[bucket.shape[0] - 1, 0])
            #print(bucket[bucket.shape[0] - 1, 1])
            bucket = np.array(list(map(
                lambda x: np.array(list(map(
                    lambda y: np.pad(y,(1,bucket_length - y.shape[0]),'constant',constant_values=(0,1)),
                    x))),
                bucket)))
            print(bucket[0, 0])
            print(bucket[0, 1])
            #print(bucket[bucket.shape[0] - 1, 0])
            #print(bucket[bucket.shape[0] - 1, 1])

            #bucket = np.array(map(np.array,bucket))
            #bucket = np.delete(bucket, np.s_[bucket_length:],axis=2) #apparently np arrays are not padded
            #print(bucket.shape)
            bucket = np.transpose(bucket,[1,2,0])
            bucket = np.split(bucket,self.bucket_size,2)
            self.buckets.append(bucket)
            self.bucket_length.append(bucket_length)


    def next_batch(self):
        x, y = self.buckets[self.bucket_pointer][self.pointer][0], self.buckets[self.bucket_pointer][self.pointer][1]
        self.pointer += 1
        if self.pointer == self.bucket_size:
            self.bucket_pointer += 1
            self.pointer = 0
        return x, y

    def random_batch(self):
        r = np.random.randint(0,self.bucket_size)
        br = np.random.randint(0, len(self.buckets))

        x, y = self.buckets[br][r][0], self.buckets[br][r][1]
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
        self.bucket_pointer = 0
