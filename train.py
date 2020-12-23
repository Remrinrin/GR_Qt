import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import AlexNet as aln
import train_data as td 

N_CLASSES = 5
IMG_W = 227
IMG_H = 227
BATCH_SIZE = 16
CAPACITY = 320
MAX_STEP = 500
learning_rate = 0.0001

train_dir = 'data/train/pose/'
#train_dir = 'data/train/photo/'

logs_train_dir = 'log/pose/'
#logs_train_dir = 'log/photo/'

train, train_label = td.get_files(train_dir)
train_batch, train_label_batch = td.get_batch(train,train_label,
                                            IMG_W,IMG_H,
                                            BATCH_SIZE,
                                            CAPACITY)

train_logits = aln.cnn_inference(train_batch, BATCH_SIZE, N_CLASSES, keep_prob=0.5)
train_loss = aln.losses(train_logits, train_label_batch)
train_op = aln.training(train_loss, learning_rate)
train__acc = aln.evaluation(train_logits, train_label_batch)

summary_op = tf.summary.merge_all()


step_list = list(range(50))
cnn_list1 = []
cnn_list2 = []

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.yaxis.grid(True)
ax.set_title('GR_accuracy ', fontsize=14, y=1.02)
ax.set_xlabel('step')
ax.set_ylabel('accuracy')
bx = fig.add_subplot(2, 1, 2)
bx.yaxis.grid(True)
bx.set_title('GR_loss ', fontsize=14, y=1.02)
bx.set_xlabel('step')
bx.set_ylabel('loss')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _op, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
            
            if step % 10 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
                cnn_list1.append(tra_acc)
                cnn_list2.append(tra_loss)
                
            if step % 50 == 0 or step == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        ax.plot(step_list, cnn_list1, color="r", label=train)
        bx.plot(step_list, cnn_list2, color="r", label=train)

        plt.tight_layout()
        #plt.show()
        plt.savefig('log/pose/train_plot.png')
        #plt.savefig('log/photo/train_plot.jpg')

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()