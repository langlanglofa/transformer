from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import codecs

from config.hyperparams import hparams as hp


def load_src_vocab():
    vocab = [line.split()[0] for line in codecs.open(hp.source_vocab, 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_tgt_vocab():
    vocab = [line.split()[0] for line in codecs.open(hp.target_vocab, 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents): 
    src2idx, _ = load_src_vocab()
    tgt2idx, _ = load_tgt_vocab()
    
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [src2idx.get(word, 1) for word in (source_sent + u" </S>").split()]
        y = [tgt2idx.get(word, 1) for word in (target_sent + u" </S>").split()] 
        if max(len(x), len(y)) <=hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)
    
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
    
    return X, Y, Sources, Targets

def load_train_data():
    src_sents = [line.strip() for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n")]
    tgt_sents = [line.strip() for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n")]
    
    X, Y, Sources, Targets = create_data(src_sents, tgt_sents)
    return X, Y
    
def load_test_data():
    src_sents = [line.strip() for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n")]
    tgt_sents = [line.strip() for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n")]
        
    X, Y, Sources, Targets = create_data(src_sents, tgt_sents)
    return X, Sources, Targets 

def get_batch_data():
    X, Y = load_train_data()
    num_batch = len(X) // hp.batch_size
    
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    input_queues = tf.train.slice_input_producer([X, Y])
            
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, num_batch 

