import tensorflow as tf
import numpy as np
from recq.recommenders.bpr import BPR
from recq.utils.data import CPRSampler
from recq.utils.inference import inner_product, mlp
from recq.utils.loss import cpr_loss


class CPR(BPR):
    def create_sampler(self, dataset, args):
        print("CPR sample!")
        self.sampler = CPRSampler(dataset, args)
    def create_mf_loss_improved(self, args):
        self.batch_pos_i = tf.placeholder(tf.int32, shape=(None,))
        batch_pos_i_embeds = tf.nn.embedding_lookup(self.i_embeds, self.batch_pos_i)

        # u_embeds: u1, u2, u1, u2, u3, ...
        u_splits = tf.split(self.batch_u_embeds, self.sampler.batch_total_sample_sizes, 0)
        i_splits = tf.split(batch_pos_i_embeds, self.sampler.batch_total_sample_sizes, 0)
        
        pos_scores = [] # 正对分数
        neg_scores = [] # 负对分数

        for idx in range(len(self.sampler.batch_total_sample_sizes)):
            # Splitting users and items embeddings into interaction and popularity parts
            u_int, u_pop = tf.split(u_splits[idx], 2, 1)
            i_int, i_pop = tf.split(i_splits[idx], 2, 1)

            u_int_list = tf.split(u_int, idx + 2, 0)
            u_pop_list = tf.split(u_pop, idx + 2, 0)
            i_int_list = tf.split(i_int, idx + 2, 0)
            i_pop_list = tf.split(i_pop, idx + 2, 0)

            if args.inference_type == "inner_product":
                pos_scores.extend(
                    [inner_product(u_i, i_i) + inner_product(u_p, i_p)
                    for u_i, i_i, u_p, i_p in zip(u_int_list, i_int_list, u_pop_list, i_pop_list)]
                )
                # For negative scores, shuffle or rotate item embeddings in each list
                neg_scores.extend(
                    [inner_product(u_i, i_i) + inner_product(u_p, i_p)
                    for u_i, i_i, u_p, i_p in zip(u_int_list, i_int_list[1:] + [i_int_list[0]], u_pop_list, i_pop_list[1:] + [i_pop_list[0]])]
                )

        # Concatenate all positive and negative scores
        pos_scores_tensor = tf.concat(pos_scores, axis=0)
        neg_scores_tensor = tf.concat(neg_scores, axis=0)

        # Compute loss based on positive and negative scores
        mf_loss = cpr_loss(pos_scores_tensor, neg_scores_tensor, args)

        # Return or store the computed loss
        self.mf_loss = mf_loss

    def create_mf_loss(self, args):
        self.batch_pos_i = tf.placeholder(tf.int32, shape=(None,))
        batch_pos_i_embeds = tf.nn.embedding_lookup(self.i_embeds, self.batch_pos_i)
        pos_scores = []
        neg_scores = []
        # u_embeds: u1, u2, u1, u2, u3, ...
        u_splits = tf.split(
            self.batch_u_embeds, self.sampler.batch_total_sample_sizes, 0
        )
        i_splits = tf.split(
            batch_pos_i_embeds, self.sampler.batch_total_sample_sizes, 0
        )
        for idx in range(len(self.sampler.batch_total_sample_sizes)):
            u_list = tf.split(u_splits[idx], idx + 2, 0)
            i_list = tf.split(i_splits[idx], idx + 2, 0)
            if args.inference_type == "inner_product":
                pos_scores.append(
                    tf.reduce_mean(
                        [inner_product(u, i) for u, i in zip(u_list, i_list)], axis=0
                    )
                )
                neg_scores.append(
                    tf.reduce_mean(
                        [
                            inner_product(u, i)
                            for u, i in zip(u_list, i_list[1:] + [i_list[0]])
                        ],
                        axis=0,
                    )
                )
            elif args.inference_type == "mlp":
                pos_scores.append(
                    tf.reduce_mean(
                        [
                            mlp(u, i, self.Ws, self.bs, self.h, args)
                            for u, i in zip(u_list, i_list)
                        ],
                        axis=0,
                    )
                )
                neg_scores.append(
                    tf.reduce_mean(
                        [
                            mlp(u, i, self.Ws, self.bs, self.h, args)
                            for u, i in zip(u_list, i_list[1:] + [i_list[0]])
                        ],
                        axis=0,
                    )
                )
        pos_scores = tf.concat(pos_scores, axis=0)
        neg_scores = tf.concat(neg_scores, axis=0)

        self.mf_loss = cpr_loss(pos_scores, neg_scores, args)

    def train_1_epoch(self, epoch):
        self.timer.start("Epoch {}".format(epoch))
        losses = []
        mf_losses = []
        reg_losses = []
        for users, items in self.sampler.sample():
            _, batch_loss, batch_mf_loss, batch_reg_loss = self.sess.run(
                [self.opt, self.loss, self.mf_loss, self.reg_loss],
                feed_dict={self.batch_u: users, self.batch_pos_i: items},
            )
            losses.append(batch_loss)
            mf_losses.append(batch_mf_loss)
            reg_losses.append(batch_reg_loss)
        self.timer.stop(
            "loss = {:.5f} = {:.5f} + {:.5f}".format(
                np.mean(losses), np.mean(mf_losses), np.mean(reg_losses)
            )
        )
