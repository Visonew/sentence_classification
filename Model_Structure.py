import numpy as np
import tensorflow as tf
import gensim
import re
from random import shuffle
import pickle

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

InputLayer1 = 300
InputLayer2 = 600
InputLayer3 = 600
HiddenLayer = 300
OutputLayer1 = 600
OutputLayer2 = 300


def make_clean_data(line):

    pat1 = re.compile(r'[？|?|！|,|.|...|0-9+|\]|\[|\(|\)|-|\'|"|-|]')
    pat2 = re.compile(r'\s+')
    data = re.sub(pat1, ' ', line)
    data = re.sub(pat2, ' ', data)
    return data


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# Encode the sentence to a fixed-length vector#
def sentence_encoder(sentence):
    word_list = sentence.split(' ')
    sen_length = len(sentence)
    words_sum = np.zeros(300)
    for i in range(len(word_list)):
        if word_list[i] in model:
            words_sum += model[word_list[i]]
        else:
            sen_length -= 1
    sentence_vec = words_sum / sen_length

    return sentence_vec


def data_process():
    with open('rt-polarity.neg', encoding='latin') as f:
        data1 = f.read()

    with open('rt-polarity.pos', encoding='latin') as f:
        data2 = f.read()

    data1 = [sentence_encoder(clean_str(sentence)) for sentence in data1.split('\n')]
    data2 = [sentence_encoder(clean_str(sentence)) for sentence in data2.split('\n')]

    pos = []
    neg = []

    for i in range(len(data1)):
        neg.append(0)

    for j in range(len(data2)):
        pos.append(1)

    all_data = data1 + data2
    tag_data = neg + pos

    index = [i for i in range(len(all_data))]
    shuffle(index)

    new_all_data = [all_data[idx] for idx in index]
    new_tag_data = [tag_data[idx] for idx in index]

    f1 = open('./new_all_data.pkl', 'wb')
    f2 = open('./new_all_tag.pkl', 'wb')

    pickle.dump(new_all_data, f1)
    pickle.dump(new_tag_data, f2)

    print('Save done!')


# Decode the fixed-length vector back to the original sentence
def decoder(input_tensor):

    with tf.variable_scope('layer1'):
        weight1 = tf.get_variable(
            'weight1', [InputLayer1, InputLayer2],
            initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
        )
        bias1 = tf.get_variable(
            'bias1', [InputLayer2],
            initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
        )

        layer1_output = tf.nn.selu(tf.matmul(input_tensor, weight1) + bias1)

    with tf.variable_scope('layer2'):
        weight2 = tf.get_variable(
            'weight2', [InputLayer2, InputLayer3],
            initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
        )
        bias2 = tf.get_variable(
            'bias2', [InputLayer3],
            initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
        )

        layer2_output = tf.nn.selu(tf.matmul(layer1_output, weight2) + bias2)

    with tf.variable_scope('layer3'):

        weight3 = tf.get_variable(
            'weight2', [InputLayer3, HiddenLayer],
            initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
        )
        bias3 = tf.get_variable(
            'bias2', [HiddenLayer],
            initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)
        )

        semantic_encoding = tf.nn.selu(tf.matmul(layer2_output, weight3) + bias3)


    return semantic_encoding


def get_batches(targets, sources, batch_size):

    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        new_targets = []
        for target in targets_batch:

            if target == 1:
                new_targets.append(model['positive'])
            else:
                new_targets.append(model['negative'])

        targets_batch = new_targets
        sources_batch = np.array(sources_batch).reshape((batch_size, 300))
        targets_batch = np.array(targets_batch).reshape((batch_size, 300))

        yield sources_batch, targets_batch


def cos_similar(x, y):
    return np.sum(np.multiply(x, y))/(np.sqrt(np.sum(np.square(x))) * (np.sqrt(np.sum(np.square(y)))))


# f1 = open('./new_all_data.pkl', 'rb')
# f2 = open('./new_all_tag.pkl', 'rb')
#
# new_all_data = pickle.load(f1)
# new_tag_data = pickle.load(f2)
#
# print(new_all_data)
# print(new_tag_data)
#
# (valid_source_batch, valid_target_batch) = next(get_batches(new_tag_data, new_all_data, 128))
# print(len(valid_source_batch))
# print(len(valid_source_batch[0]))
# print(len(valid_target_batch))
# print(len(valid_target_batch[0]))
