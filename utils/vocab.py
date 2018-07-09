"""
Vocabulary
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import codecs

ID_MATRIX = list()

def load_vocab(filename):
    vocab = []
    with tf.gfile.GFile(filename) as fd:
        for line in fd:
            word = line.strip()
            vocab.append(word)

    return vocab
        
def load_entity(batch_size, max_length):
    ENTITY_PATH = '../corpus/dict.txt'
    vocab = load_vocab('../corpus/vocab.txt')
    ids = list()
    with codecs.open(ENTITY_PATH, 'r', 'utf-8') as f:
        entity_list = [w.strip() for w in f.readlines()]
    for e in entity_list:
        if e in vocab:
            ids.append(vocab.index(e))
    for i in range(batch_size * max_length):
        ID_MATRIX.append([1 if i in ids else 0 for i in range(len(vocab))])