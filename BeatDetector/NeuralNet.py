import tensorflow as tf
import os
from random import randint

class NeuralNet:
    
    def __init__(self):
        self.inputData = None
        self.expectedClasses = None
        self.inputLengths = None
        self.cross_entropy = None
        self.trainer = None
        self.accuracy = None
        self.predictionMax = None
        self.classifier = None
        self.sess = tf.Session()
        self.make_new()
        self.step = 0

    def make_new(self):
        maxLen = 200
        exampleLen = 120
        numMemCells = 20
        self.inputData = tf.placeholder(tf.float32, [None, maxLen, exampleLen], name="examples")
        self.expectedClasses = tf.placeholder(tf.float32, [None, maxLen, 2], name="labels")
        self.inputLengths = tf.placeholder(tf.int32, [None], name="lengths")
        #print("########################")
        #print(tf.shape(self.inputData)[0])
                
        cell = tf.nn.rnn_cell.LSTMCell(numMemCells, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2, state_is_tuple=True)
        cellOut, cellState = tf.nn.dynamic_rnn(
            cell, self.inputData, dtype=tf.float32, sequence_length=self.inputLengths)
        
        #last = cellState[1]
        #cellOut = tf.transpose(cellOut, [1, 0, 2])
        #batchSize = tf.shape(cellOut)[0]
        #index = tf.range(0, batchSize) * maxLen + (self.inputLengths - 1)
        #flat = tf.reshape(cellOut, [-1, self.numMemCells])
        #last = tf.gather(flat, index)
        #print(last.get_shape())
        #last = tf.gather(cellOutTrans, int(cellOutTrans.get_shape()[0]) - 1)
        
        weight = tf.Variable(tf.truncated_normal([numMemCells, 2], stddev = 0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[2]))
        #prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        #prediction = tf.matmul(cellOut, weight) + bias
        flat = tf.reshape(cellOut, (-1, numMemCells))
        prediction = tf.matmul(flat, weight) + bias
        prediction = tf.reshape(prediction, (tf.shape(self.inputData)[0], 200, 2))

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, self.expectedClasses), name="cross_entropy")
        #cross_entropy = -tf.reduce_sum(self.expectedClasses * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
        optimizer = tf.train.AdamOptimizer()
        self.trainer = optimizer.minimize(self.cross_entropy, name="trainer")
        
        correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(self.expectedClasses,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        self.predictionMax = tf.argmax(prediction, 1)
        self.classifier = tf.nn.softmax(prediction)
        
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)


    def train(self, examples, labels, batch_size, iterations):
        lengths = [200] * batch_size
        
        feed = {self.inputData: examples,
                self.expectedClasses: labels,
                self.inputLengths: lengths}
        
        for i in range(iterations):
            self.sess.run(self.trainer, feed)

        index = randint(0, batch_size - 1)
        feed = {
            self.inputData: [examples[index]],
            self.expectedClasses: [labels[index]],
            self.inputLengths: lengths[:1]
        }
        ret = self.sess.run(self.classifier, feed)
        return ret[0, :, 1], labels[index][:, 1]
                
    save_dir = "./save/"
    def save(self):
        if not os.path.exists(NeuralNet.save_dir):
            os.makedirs(NeuralNet.save_dir)
        saver = tf.train.Saver()
        saver.save(self.sess, NeuralNet.save_dir + "model", global_step=self.step)
        self.step += 1

    def load(self):
        #saver = tf.train.import_meta_graph(NeuralNet.save_dir + "model.meta")
        saver = tf.train.Saver()      
        saver.restore(self.sess, tf.train.latest_checkpoint(NeuralNet.save_dir))