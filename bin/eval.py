from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import codecs
import os

from nltk.translate.bleu_score import corpus_bleu
from config.hyperparams import hparams as hp
from transformer.feeder import load_test_data, load_src_vocab, load_tgt_vocab
from transformer.model import Transformer


def eval(): 
    g = Transformer(is_training=False)
    
    X, Sources, Targets = load_test_data()
    src2idx, idx2src = load_src_vocab()
    tgt2idx, idx2tgt = load_tgt_vocab()

    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
              
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] 
             
            if not os.path.exists('results'): 
                os.mkdir('results')
            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in range(len(X) // hp.batch_size):
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                    targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                     
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]
                     
                    for source, target, pred in zip(sources, targets, preds): 
                        got = " ".join(idx2tgt[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write("- source: " + source +"\n")
                        fout.write("- expected: " + target + "\n")
                        fout.write("- got: " + got + "\n\n")
                        fout.flush()
                          
                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)
              
                score = corpus_bleu(list_of_refs, hypotheses)
                print ("Bleu Score = " + str(100*score))
                                          
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("--gpu_index", default=0, type=int,
                            help="GPU index.")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    eval()