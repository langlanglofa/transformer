from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 


hparams = tf.contrib.training.HParams (

    # data
    source_train = '../corpus/wmt/en-de/train.de',
    target_train = '../corpus/wmt/en-de/train.en',
    source_test = '../corpus/wmt/en-de/newstest2015.de',
    target_test = '../corpus/wmt/en-de/newstest2015.en',
    source_vocab = '../corpus/wmt/en-de/vocab.de',
    target_vocab = '../corpus/wmt/en-de/vocab.en',
    
    # training
    batch_size = 32,
    lr = 0.0001,
    logdir = 'log',
    
    # model
    src_vocab_size = 50000,
    tgt_vocab_size = 50000,
    maxlen = 30,
    min_cnt = 2, 
    hidden_units = 512,
    num_blocks = 6,
    num_epochs = 20,
    num_heads = 8,
    dropout_rate = 0.1,
    label_smooth = 0.1,
    sinusoid = True # If True, use sinusoid. If false, positional embedding.

)