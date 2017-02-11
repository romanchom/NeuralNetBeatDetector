import tensorflow as tf
import os
from random import randint
import numpy as np
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from Config import Config

numMemCells = 50

class NeuralNet:
    
    def __init__(self):
        self.sess = tf.Session()
        self.make_new()
        self.step = 0

    def make_variables(self):
        
        # MODEL VARIABLES   
        with tf.variable_scope('variables'):     
            # recurent cells
            self.cell = tf.nn.rnn_cell.LSTMCell(numMemCells, state_is_tuple=True)
            self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * 3, state_is_tuple=True)
            
            # projection matrix
            self.weight = tf.Variable(tf.truncated_normal([numMemCells, 2], stddev = 0.1))
            self.bias = tf.Variable(tf.constant(0.1, shape=[2]))

    def make_training_graph(self):
        # TRAINING GRAPH
        
        with tf.variable_scope('training'):
        # INPUT DATA
        
            with tf.variable_scope('input'):
                self.examples = tf.placeholder(tf.float32, [None, Config.framesPerExample, Config.exampleLength], name="examples")
                self.labels = tf.placeholder(tf.float32, [None, Config.framesPerExample, 2], name="labels")
                self.inputLengths = tf.placeholder(tf.int32, [None], name="lengths")
                
            
            with tf.variable_scope('operations'):
                # TRAINING AND VALIDATION OPERATIONS
                cell_out, cell_state = tf.nn.dynamic_rnn(
                    self.cell, self.examples, dtype=tf.float32, sequence_length=self.inputLengths)
            
                flat = tf.reshape(cell_out, (-1, numMemCells))
                prediction = tf.matmul(flat, self.weight) + self.bias
                prediction = tf.reshape(prediction, (tf.shape(self.examples)[0], Config.framesPerExample, 2))
                
                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(prediction, self.labels)
                self.cross_entropy = tf.slice(self.cross_entropy, [0, Config.framesPerExample // 2], [-1, -1])
                print(self.cross_entropy.get_shape())
                self.cross_entropy = tf.reduce_mean(self.cross_entropy, name="cross_entropy")
                
                optimizer = tf.train.AdamOptimizer()
                self.optimize = optimizer.minimize(self.cross_entropy, name="optimize")
        
                # EXAMINATION OPERATION
                self.prediction = tf.nn.softmax(prediction)

    def assign_state_op(self, variables, state, name):
        update_ops = []
        for state_variable, new_state in zip(variables, state):
            update_ops.append(state_variable[0].assign(new_state[0]))
            update_ops.append(state_variable[1].assign(new_state[1]))
        return tf.tuple(update_ops, name=name)

    def make_inference_graph(self):
        #INFERENCE GRAPH
        
        with tf.variable_scope('inference') as inference_scope:
            #INFERENCE INPUT - single "batch"
            self.inference_input = tf.placeholder(tf.float32, [1, Config.exampleLength], name="input")
            self.input_full_name = self.inference_input.op.name
            
            # LSTM cell state variables
            state_variables = []
            for state_c, state_h in self.cell.zero_state(1, tf.float32):
                state_variables.append(tf.nn.rnn_cell.LSTMStateTuple(
                    tf.Variable(state_c, trainable=False, name="cell_state"),
                    tf.Variable(state_h, trainable=False, name="hidden_state")))
            cell_state_variables = tuple(state_variables)

        # INFERENCE OPERATION
        # we need to use such wild scope, because we need to share
        # variables with training model
        # tensorflow is weird and variables are created each time
        # a cell is called, each time in possibly different scope
        with tf.variable_scope('training/operations/RNN') as scope:
            scope.reuse_variables()
            cell_out, cell_state = self.cell(self.inference_input, cell_state_variables)


        with tf.variable_scope('inference/'):
            # LSTM cell state update operation, needs to be run, for cell state to change
            self.cell_state_update_op = self.assign_state_op(state_variables, cell_state, "cell_state_update_op")

            with tf.control_dependencies(self.cell_state_update_op):
                flat = tf.reshape(cell_out, [1, numMemCells])
                projection = tf.matmul(flat, self.weight) + self.bias
                self.inference_op = tf.nn.softmax(projection, name="output")
                self.output_full_name = self.inference_op.op.name

            
            # RESET OP
            self.reset_cell_state_op = self.assign_state_op(state_variables, self.cell.zero_state(1, tf.float32), "reset_op")
            cell_vars = [elem for t0 in state_variables for elem in t0]
            self.var_init_full_name = tf.variables_initializer(cell_vars, name="init").name
            


    def make_new(self):
        self.make_variables()
        self.make_training_graph()
        self.make_inference_graph()
        
        init_op = tf.global_variables_initializer()
        print(init_op.name)
        self.sess.run(init_op)
        
        writer = tf.summary.FileWriter("./log", self.sess.graph)
        writer.close()
       

    def make_feed(self, examples, labels):
        return {
            self.examples: examples,
            self.labels: labels,
            self.inputLengths: [Config.framesPerExample] * len(examples)
        }

    def train(self, examples, labels):
        feed = self.make_feed(examples, labels)
        self.sess.run(self.optimize, feed)

    def validate(self, examples, labels):
        feed = self.make_feed(examples, labels)
        return self.sess.run(self.cross_entropy, feed)

    def examine(self, example):
        feed = {
            self.examples: [example],
            self.inputLengths: [Config.framesPerExample]
        }

        return self.sess.run(self.prediction, feed)[0]
    
    def single_step_examine(self, example):
        feed = {}
        output = np.zeros([Config.framesPerExample])
        for i in range(len(example)):
            feed[self.inference_input] = [example[i]]
            ret = self.sess.run([self.inference_op], feed)
            output[i] = ret[0][0, 1]
        return np.array(output)
                
    save_dir = "./save/"
    def save(self):
        if not os.path.exists(NeuralNet.save_dir):
            os.makedirs(NeuralNet.save_dir)
        saver = tf.train.Saver()
        saver.save(self.sess, NeuralNet.save_dir + "model", global_step=self.step)
        self.step += 1
    
    def export_to_protobuffer(self, directory):
        whitelist = [v.op.name for v in tf.trainable_variables()]
        inference_graph = convert_variables_to_constants(self.sess, self.sess.graph_def, [self.output_full_name, self.var_init_full_name], whitelist)
        
        tf.train.write_graph(inference_graph, directory, 'minimal_graph.pb', as_text=False)
        tf.train.write_graph(inference_graph, directory, 'minimal_graph.pbtxt', as_text=True)
        print("Graph saved")
        print("Use \"{}\" as input name".format(self.input_full_name))
        print("Use \"{}\" as output name".format(self.output_full_name))
        print("Use \"{}\" as init name".format(self.var_init_full_name))

    def load(self):
        #saver = tf.train.import_meta_graph(NeuralNet.save_dir + "model.meta")
        saver = tf.train.Saver()      
        saver.restore(self.sess, tf.train.latest_checkpoint(NeuralNet.save_dir))