import tensorflow as tf
import Model_Structure as MS
import pickle
import numpy as np

numbda = 16
batch_size = 256

f1 = open('./new_all_data.pkl', 'rb')
f2 = open('./new_all_tag.pkl', 'rb')

new_all_data = pickle.load(f1)
new_tag_data = pickle.load(f2)

train_graph = tf.Graph()

tag1 = MS.model['positive']
tag2 = MS.model['negative']
tag1 = np.reshape(tag1, newshape=(1, 300))
tag2 = np.reshape(tag2, newshape=(1, 300))

with train_graph.as_default():

    input_data = tf.placeholder(dtype=tf.float32, shape=[None, None], name='input_data')
    output_data = tf.placeholder(dtype=tf.float32, shape=[None, None], name='output_data')
    input_tag = tf.placeholder(dtype=tf.float32, shape=[None, None], name='input_tag')

    semantic_state, output = MS.decoder(input_data)

    semantic_state = tf.identity(semantic_state, name='semantic_state')

    output = tf.identity(output, name='output')

    with tf.name_scope('optimization'):  # Loss function #
        cost = tf.reduce_mean(numbda * tf.pow(semantic_state - input_tag, 2))
               # + tf.reduce_mean(tf.pow(output - output_data, 2))

        optimizer = tf.train.AdamOptimizer(0.003)
        gradients1 = optimizer.compute_gradients(cost)
        train_op1 = optimizer.apply_gradients(gradients1)


train_source = new_all_data[batch_size:]
train_target = new_tag_data[batch_size:]

valid_source = new_all_data[:batch_size]
valid_target = new_tag_data[:batch_size]

(valid_source_batch, valid_target_batch) = next(MS.get_batches(valid_target, valid_source, batch_size))

display_step = 100  
checkpoint = './data/trained_model.ckpt'

loaded_graph = tf.Graph()
epochs = 200
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(1, epochs + 1):
        for batch_i, (train_source_batch, train_target_batch) in enumerate(
            MS.get_batches(train_target, train_source, batch_size)
        ):
            _, loss = sess.run(
                [train_op1, cost], feed_dict={
                    input_data: train_source_batch,
                    input_tag: train_target_batch,
                    output_data: train_source_batch,
                }
            )

            if batch_i % display_step == 0:
                validation_loss = sess.run(
                    [cost], feed_dict={
                        input_data: valid_source_batch,
                        input_tag: valid_target_batch,
                        output_data: valid_source_batch
                    }
                )

                print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                      .format(epoch_i,
                              epochs,
                              batch_i,
                              len(train_source) // batch_size,
                              loss,
                              validation_loss[0]))

                answer1 = sess.run(semantic_state, {input_data: train_source})
                answer2 = np.array([])

                count = 0
                for sentenceVec in answer1:
                    if MS.cos_similar(sentenceVec, tag1) > MS.cos_similar(sentenceVec, tag2):
                        answer2 = np.append(answer2, [1])
                    else:
                        answer2 = np.append(answer2, [0])

                for i in range(len(answer2)):
                    if answer2[i] == train_target[i]:
                        count += 1

                train_source_accuracy = count/(len(train_source))

                answer1 = sess.run(semantic_state, {input_data: valid_source})
                answer2 = np.array([])

                count = 0
                for sentenceVec in answer1:
                    if MS.cos_similar(sentenceVec, tag1) > MS.cos_similar(sentenceVec, tag2):
                        answer2 = np.append(answer2, [1])
                    else:
                        answer2 = np.append(answer2, [0])

                for i in range(len(answer2)):
                    if answer2[i] == valid_target[i]:
                        count += 1

                valid_source_accuracy = count/(len(valid_source))

                print('train_accuracy: ', train_source_accuracy, 'valid_accuracy: ', valid_source_accuracy)

    saver = tf.train.Saver()
    saver.save(sess, checkpoint)
    print('Model Trained and Saved')






