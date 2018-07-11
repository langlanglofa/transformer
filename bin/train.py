from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import os
import tqdm

from config.hyperparams import hparams as hp
from transformer.feeder import get_batch_data, load_src_vocab, load_tgt_vocab
from transformer.model import Transformer


if __name__ == '__main__':      
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("--gpu_index", default=0, type=int,
                            help="GPU index.")
    args = parser.parse_args()

    g = Transformer()
    
    sv = tf.train.Supervisor(graph=g.graph, 
                             logdir=hp.logdir,
                             save_model_secs=0)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs+1): 
            if sv.should_stop():
                break
            for step in tqdm.tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)
                
            gs = sess.run(g.global_step)   
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))