import cv2
import AlexNet as aln
import train_data as td 
import tensorflow as tf
from PIL import Image
from GetPose import *

def GRStart(img_ROI):
    path_photo = 'data/testImage/photo/test_photo.jpg'
    cv2.imwrite(path_photo ,img_ROI)
    frame_pose = GetPose(path_photo)
    train_dir = 'data/testImage/pose/'
    cv2.imwrite(train_dir + 'test_pose.jpg', frame_pose)

    train, train_label = td.get_files(train_dir)

    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]
    image = Image.open(img_dir)
    image = image.resize([227, 227])
    image_array = np.array(image)    
    
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 5

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 227, 227, 3])
        logit = aln.cnn_inference(image, BATCH_SIZE, N_CLASSES, keep_prob=1)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[227, 227, 3])

        logs_train_dir = 'log/pose/'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                return 1
            elif max_index == 1:
                return 2
            elif max_index == 2:
                return 3
            elif max_index == 3:
                return 4
            elif max_index == 4:
                return 5

#frame = cv2.imread('data/testImage/pose/test_pose.jpg')
#GRStart(frame)