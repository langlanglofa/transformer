from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import codecs
import os

from collections import Counter
from config.hyperparams import hparams as hp


def make_vocab(fpath, fname, vocab_size):
    text = codecs.open(fpath, 'r', 'utf-8').read()
    words = text.split()
    word2cnt = Counter(words)
    
    with codecs.open(fname, 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(vocab_size-4):
            fout.write(u"{}\t{}\n".format(word, cnt))

if __name__ == '__main__':
    make_vocab(hp.source_train, hp.source_vocab, hp.src_vocab_size)
    make_vocab(hp.target_train, hp.target_vocab, hp.tgt_vocab_size)