import os
import numpy as np
import tensorflow as tf

def get_files(file_dir):
    gesture_1 = []
    label_gesture_1 = []
    gesture_2 = []
    label_gesture_2 = []
    gesture_3 = []
    label_gesture_3 = []
    gesture_4 = []
    label_gesture_4 = []
    gesture_5 = []
    label_gesture_5 = []

    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'gesture_1':
            gesture_1.append(file_dir + file)
            label_gesture_1.append(0)
        elif name[0] == 'gesture_2':
            gesture_2.append(file_dir + file)
            label_gesture_2.append(1)
        elif name[0] == 'gesture_3':
            gesture_3.append(file_dir + file)
            label_gesture_3.append(2)
        elif name[0] == 'gesture_4':
            gesture_4.append(file_dir + file)
            label_gesture_4.append(3)
        else:
            gesture_5.append(file_dir + file)
            label_gesture_5.append(4)

    image_list = np.hstack((gesture_1, gesture_2, gesture_3, gesture_4, gesture_5))
    label_list = np.hstack((label_gesture_1, label_gesture_2, label_gesture_3, label_gesture_4, label_gesture_5))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    np.random.shuffle(temp)
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])

    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    label = input_queue[1]

    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    image_batch = tf.cast(image_batch, tf.float32)
    label_batch = tf.cast(label_batch, tf.int32)

    return image_batch, label_batch