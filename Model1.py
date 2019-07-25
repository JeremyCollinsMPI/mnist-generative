# Bayesian image classifier, which classifies images from MNIST with 10 images per class as training data 
# now incorporating something like convolutional filters

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
import numpy as np
import tensorflow as tf
import random
import scipy
import scipy.stats
tf.logging.set_verbosity(tf.logging.INFO)



# set random seed
random.seed(13)

# set training data size
training_data_size = 10


# Load data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32) 


def copy(x):
	new=[]
	for member in x:
		new.append(member)
	return new	
	
def rep(x,y):
	new=[]
	for m in xrange(y):
		new.append(x)
	return new
	
def replist(x,y):
	new=[]
	for m in xrange(y):
		new.append(copy(x))
	return new

def show(number):
	x = train_data[number]
	x = [1-y for y in x] 
	img = Image.new('1',(28,28))
	img.putdata(x)
	img.show()

def showImage(x, size = 784):
	x = np.reshape(x,size)
	x = [1-y for y in x] 
	img = Image.new('1',(int(size**0.5),int(size**0.5)))
	img.putdata(x)
	img.show()

def log(x):
	return np.log(x)
	

def returnImage(imageNumber1, dataset = train_data):
	first = dataset[imageNumber1]
	first = np.reshape(first, (28,28))
	return first

classes = {}
for number in xrange(10):
	classes[number] = [x for x in xrange(len(train_labels)) if train_labels[x] == number]
training_images = {}
for number in xrange(10):
	for m in xrange(training_data_size):
		training_images[str(number)+'_'+str(m)] = returnImage(random.sample(classes[number],1)[0])


# define the pizel size of elements
size = [5,5]
radius = int((size[0]-1)/2)

def findLocalRegion(image, x, y, padding = True):
	radius = int(size[0]/2 - 0.5)
	if padding:
		image = np.pad(image, ((radius, radius), (radius, radius)), 'constant', constant_values = (0,0))
		x = x + radius
		y = y + radius
	# croppedImage = image[max(0,x-radius):min(32,x+radius+1), max(0,y-radius):min(32,y+radius+1)]
	croppedImage = image[(x-radius):(x+radius+1), (y-radius):(y+radius+1)]
	return croppedImage

class FirstModel:

    def __init__(self):
        self.learn_rate = 0.06
        self.datapoint_size = 1000
        self.batch_size = self.datapoint_size
        self.target = np.reshape(train_data, (55000, 28, 28, 1))
        number_of_filters = 10
        self.steps = 100000
        self.inputs = tf.Variable(tf.random_uniform([self.batch_size, number_of_filters], minval = 0, maxval = 1, seed = 10))
        self.input_maxima = tf.reduce_max(self.inputs, 1, keepdims = True)
        self.compare_inputs_with_maxima = tf.equal(self.inputs, self.input_maxima)
        self.rounded_inputs = tf.cast(self.compare_inputs_with_maxima, tf.float32)
        self.layer = tf.Variable(tf.random_uniform([number_of_filters, 28, 28, 1], minval = 0, maxval = 1, seed = 10))
#         self.layer_reshaped = tf.reshape(self.layer, (number_of_filters, 28 * 28))
#         self.y = tf.matmul(self.inputs, self.layer)
        self.y = tf.tensordot(self.rounded_inputs, self.layer, axes = [1, 0])
        self.y = tf.reshape(self.y, (-1, 28 * 28))
        self.y_ = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y_reshaped = tf.reshape(self.y_, (-1, 28 * 28))
        self.cost = tf.reduce_mean(tf.square(self.y_reshaped - self.y))
        self.train_step = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(self.cost)
        self.clip_op1 = tf.assign(self.inputs, tf.clip_by_value(self.inputs, 0, np.infty))
        self.clip_op2 = tf.assign(self.layer, tf.clip_by_value(self.layer, 0, np.infty))
        self.input_sums = tf.reduce_sum(self.inputs, 1, keepdims = True)
        self.normalise_inputs = tf.assign(self.inputs, self.inputs / self.input_sums)
        
        
        
        
        
#             mean = tf.reduce_mean(self.one_hot_encodings_variables[str(i)])
#             self.reduction_ops[str(i)] = tf.assign(self.one_hot_encodings_variables[str(i)], self.one_hot_encodings_variables[str(i)] - mean)

        
    def train(self):
        batch_size = self.batch_size
        datapoint_size = self.datapoint_size
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        for i in xrange(self.steps):
            print(i)
            if datapoint_size == batch_size:
                batch_start_idx = 0
            elif datapoint_size < batch_size:
                raise ValueError("datapoint_size: %d, must be greater than batch_size: %d" % (datapoint_size, batch_size))
            else:
                batch_start_idx = (i * batch_size) % (datapoint_size - batch_size)
            batch_end_idx = batch_start_idx + batch_size
            batch_y = self.target[batch_start_idx:batch_end_idx]
            feed = {self.y_: batch_y}
            sess.run(self.train_step, feed_dict = feed)
            sess.run(self.clip_op1)
            sess.run(self.clip_op2)
            sess.run(self.normalise_inputs)
            print("After %d iterations:" % i)
#             print("cost: " % sess.run(self.cost, feed_dict = feed))
            print("y: %s" % sess.run(self.y_reshaped, feed_dict = feed))
            print("cost: %s" % sess.run(self.cost, feed_dict = feed))
            print("filter: %s" % sess.run(self.layer))
            print(("inputs: %s" % sess.run(self.inputs)))
#             if self.include_intercept:
#                 print("b: %f" % sess.run(self.b))
#             for j in xrange(len(self.one_hot_encodings)):
#                 print("one_hot_encodings_variable" + str(j) + " : %s" % sess.run(self.one_hot_encodings_variables[str(j)]))


class SecondModel:

    def __init__(self):
        self.learn_rate = 0.06
        self.datapoint_size = 1000
        self.batch_size = self.datapoint_size
        self.target = np.reshape(train_data, (55000, 28, 28, 1))
        number_of_filters = 10
        self.steps = 100000
        self.inputs = tf.Variable(tf.random_uniform([self.batch_size, number_of_filters], minval = 0, maxval = 1, seed = 10))
        self.input_maxima = tf.reduce_max(self.inputs, 1, keepdims = True)
        self.compare_inputs_with_maxima = tf.equal(self.inputs, self.input_maxima)
        self.rounded_inputs = tf.cast(self.compare_inputs_with_maxima, tf.float32)
        self.layer = tf.Variable(tf.random_uniform([number_of_filters, 28, 28, 1], minval = 0, maxval = 1, seed = 10))
#         self.layer_reshaped = tf.reshape(self.layer, (number_of_filters, 28 * 28))
#         self.y = tf.matmul(self.inputs, self.layer)
        self.y = tf.tensordot(self.rounded_inputs, self.layer, axes = [1, 0])
        self.y = tf.reshape(self.y, (-1, 28 * 28))
        self.y_ = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y_reshaped = tf.reshape(self.y_, (-1, 28 * 28))
        self.cost = tf.reduce_mean(tf.square(self.y_reshaped - self.y))
        self.train_step = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(self.cost)
        self.clip_op1 = tf.assign(self.inputs, tf.clip_by_value(self.inputs, 0, np.infty))
        self.clip_op2 = tf.assign(self.layer, tf.clip_by_value(self.layer, 0, np.infty))
        self.input_sums = tf.reduce_sum(self.inputs, 1, keepdims = True)
        self.normalise_inputs = tf.assign(self.inputs, self.inputs / self.input_sums)
        
        
        
        
        
#             mean = tf.reduce_mean(self.one_hot_encodings_variables[str(i)])
#             self.reduction_ops[str(i)] = tf.assign(self.one_hot_encodings_variables[str(i)], self.one_hot_encodings_variables[str(i)] - mean)

        
    def train(self):
        batch_size = self.batch_size
        datapoint_size = self.datapoint_size
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        for i in xrange(self.steps):
            print(i)
            if datapoint_size == batch_size:
                batch_start_idx = 0
            elif datapoint_size < batch_size:
                raise ValueError("datapoint_size: %d, must be greater than batch_size: %d" % (datapoint_size, batch_size))
            else:
                batch_start_idx = (i * batch_size) % (datapoint_size - batch_size)
            batch_end_idx = batch_start_idx + batch_size
            batch_y = self.target[batch_start_idx:batch_end_idx]
            feed = {self.y_: batch_y}
            sess.run(self.train_step, feed_dict = feed)
            sess.run(self.clip_op1)
            sess.run(self.clip_op2)
            sess.run(self.normalise_inputs)
            print("After %d iterations:" % i)
#             print("cost: " % sess.run(self.cost, feed_dict = feed))
            print("y: %s" % sess.run(self.y_reshaped, feed_dict = feed))
            print("cost: %s" % sess.run(self.cost, feed_dict = feed))
            print("filter: %s" % sess.run(self.layer))
            print(("inputs: %s" % sess.run(self.inputs)))
#             if self.include_intercept:
#                 print("b: %f" % sess.run(self.b))
#             for j in xrange(len(self.one_hot_encodings)):
#                 print("one_hot_encodings_variable" + str(j) + " : %s" % sess.run(self.one_hot_encodings_variables[str(j)]))



# class FirstModel



image = returnImage(1)
print(np.shape(image))
showImage(image)
# train_data = np.reshape(train_data, (55000, 28, 28, 1))
# print(np.shape(train_data))

model = FirstModel()
model.train()


# self.cnn_input_layer = tf.placeholder(tf.float32, [len(self.dependent_variables), len(self.dependent_variables[0])])
# self.cnn_input_layer_reshaped = tf.reshape(self.cnn_input_layer, [-1, len(self.dependent_variables[0]), 1])
# self.cnn_conv1_filters = 5
# self.cnn_conv1 = tf.layers.conv1d(self.cnn_input_layer_reshaped, filters = self.cnn_conv1_filters, kernel_size = 2, padding = "same", activation = tf.nn.relu)
# self.cnn_last = self.cnn_conv1
# number_of_cnn_outputs = 1
# cnn_last_shape = tf.shape(self.cnn_last)
# cnn_output_length = len(self.dependent_variables[0]) * self.cnn_conv1_filters   
# self.cnn_last_flat = tf.reshape(self.cnn_last, [-1, cnn_output_length])
# self.cnn_output = tf.layers.dense(self.cnn_last_flat, units = number_of_cnn_outputs, activation = tf.nn.sigmoid)
# self.cnn_coefficients =  tf.Variable(tf.zeros([number_of_cnn_outputs, 1]))
# self.cnn_to_add = tf.matmul(self.cnn_output, self.cnn_coefficients)
# self.cnn_to_add = self.cnn_to_add - tf.reduce_mean(self.cnn_to_add)
# self.y = self.y + self.cnn_to_add 
# 
# 
# 
# 
# 
# import numpy as np
# import tensorflow as tf
# import time
# from general import *
# from OneHotEncoder import OneHotEncoder
# 
# class LinearRegression:
#     steps = 1000000
#     learn_rate = 0.03
#     def __init__(self, target, dependent_variables, positive = True, include_intercept = False):
#         self.target = np.array(target)
#         self.dependent_variables = np.array(dependent_variables)
#         self.positive = positive
#         self.include_intercept = include_intercept
#         self.one_hot_encodings = []	
#         self.other_vectors = []
# 
#     def include_one_hot_encoding(self, one_hot_encoding):
#         self.one_hot_encodings.append(one_hot_encoding)
#     
#     def create_linear_layers(self):
#         self.x = tf.placeholder(tf.float32, [None, len(self.dependent_variables[0])], name = "x")
#         self.W = tf.Variable(tf.zeros([len(self.dependent_variables[0]), 1]), name = "W")
#         if self.include_intercept:
#             self.b = tf.Variable(tf.zeros([1]), name = "b")
#         self.product = tf.matmul(self.x, self.W)
#         if self.include_intercept:
#             self.y = self.product + self.b
#         else:
#             self.y = self.product
#    
#     def create_one_hot_layers(self):
#         if len(self.one_hot_encodings) > 0:
#             self.one_hot_encodings_placeholders = {}
#             self.one_hot_encodings_encodings = {}
#             self.one_hot_encodings_variables = {}
#             for i in xrange(len(self.one_hot_encodings)):
#                 number_of_categories = self.one_hot_encodings[i].depth
#                 self.one_hot_encodings_placeholders[str(i)] = tf.placeholder(tf.int32, [None], name = "one_hot_placeholder" + str(i))
#                 self.one_hot_encodings_encodings[str(i)] = tf.one_hot(self.one_hot_encodings_placeholders[str(i)], number_of_categories)
#                 self.one_hot_encodings_variables[str(i)] = tf.Variable(tf.zeros([number_of_categories, 1], name = "one_hot_variable" + str(i)))  
#         for i in xrange(len(self.one_hot_encodings)):
#             print self.one_hot_encodings_encodings[str(i)]
#             self.y = self.y + tf.matmul(self.one_hot_encodings_encodings[str(i)], self.one_hot_encodings_variables[str(i)])
# 
#     def create_other_vector_layers(self):
#         if len(self.other_vectors) > 0:
#             self.other_vectors_placeholders = {}
#             self.other_vectors_variables = {}
#             for i in xrange(len(self.other_vectors)):
#                 self.other_vectors_placeholders[str(i)] = tf.placeholder(tf.float32, [1, None], name = "other_vectors_placeholder" + str(i))
#                 self.other_vectors_variables[str(i)] = tf.Variable(tf.zeros([None, 1]), name = "other_vectors_variable" + str(i))
#         for i in xrange(len(self.other_vectors)):
#             self.y = self.y + tf.matmul(self.other_vectors_placeholders[str(i)], self.other_vectors_variables[str(i)])
# 
#     def train(self):      
#         datapoint_size = len(self.target)
#         batch_size = datapoint_size
#         self.create_linear_layers()
#         self.create_one_hot_layers()        
#         self.y_ = tf.placeholder(tf.float32, [None, 1])
#         self.cost = tf.reduce_mean(tf.square(self.y_ - self.y))
#         self.cost_sum = tf.summary.scalar("cost", self.cost)
#         self.train_step = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(self.cost)	
#         self.clip_op = tf.assign(self.W, tf.clip_by_value(self.W, 0, np.infty))
#         self.reduction_ops = {}
#         for i in xrange(len(self.one_hot_encodings)):
#             mean = tf.reduce_mean(self.one_hot_encodings_variables[str(i)])
#             self.reduction_ops[str(i)] = tf.assign(self.one_hot_encodings_variables[str(i)], self.one_hot_encodings_variables[str(i)] - mean)
#         sess = tf.Session()
#         init = tf.initialize_all_variables()
#         sess.run(init)
#         for i in xrange(self.steps):
#             print i
#             if datapoint_size == batch_size:
#                 batch_start_idx = 0
#             elif datapoint_size < batch_size:
#                 raise ValueError("datapoint_size: %d, must be greater than batch_size: %d" % (datapoint_size, batch_size))
#             else:
#                 batch_start_idx = (i * batch_size) % (datapoint_size - batch_size)
#             batch_end_idx = batch_start_idx + batch_size
#             batch_x = self.dependent_variables[batch_start_idx:batch_end_idx]
#             batch_y = self.target[batch_start_idx:batch_end_idx]
#             feed = {self.x: batch_x, self.y_: batch_y}
#             for j in xrange(len(self.one_hot_encodings)):
#                 to_feed = self.one_hot_encodings[j].encoding[batch_start_idx:batch_end_idx]
#                 feed[self.one_hot_encodings_placeholders[str(j)]] = to_feed
#             sess.run(self.train_step, feed_dict = feed)
#             if self.positive:
#                 sess.run(self.clip_op)
#             for j in xrange(len(self.one_hot_encodings)):
#             	sess.run(self.reduction_ops[str(j)])
#             print("After %d iteration:" % i)
#             print("W: %s" % sess.run(self.W))
#             if self.include_intercept:
#                 print("b: %f" % sess.run(self.b))
#             for j in xrange(len(self.one_hot_encodings)):
#                 print("one_hot_encodings_variable" + str(j) + " : %s" % sess.run(self.one_hot_encodings_variables[str(j)]))
# 
# class SequenceLinearRegression(LinearRegression):
# 	def __init__(self, sequences, window_size, offset = 0, positive = True, include_intercept = False, use_padding = True):
# 		self.window_size = window_size
# 		self.offset = offset
# 		self.sequences = sequences
# 		self.positive = positive
# 		self.include_intercept = include_intercept
# 		dependent_variables = []
# 		target_variable = []
# 		for sequence in self.sequences:
# 			normalised_sequence = []
# 			for n in range(1, len(sequence)):
# 				normalised_sequence.append([sequence[n] / sequence[n-1]])
# 				to_append = []
# 				for number in range(n - window_size, n - offset):
# 					if number < 0: 
# 						if use_padding == True:
# 							to_append.append(sequence[0])
# 					else:
# 						to_append.append(sequence[number])
# 				normaliser = to_append[-1]
# 				to_append = np.array(to_append)				
# 				to_append = to_append / normaliser
# 				dependent_variables.append(to_append)
# 			target_variable = target_variable + normalised_sequence
# 		self.target = np.array(target_variable)
# 		self.dependent_variables = np.array(dependent_variables)
# 		self.one_hot_encodings = []
# 
# class SequenceLinearRegressionIncludingTimestamps(SequenceLinearRegression):
#     def include_timestamps(self, timestamps):    
#         self.timestamps = timestamps
#         self.calendar_months = [find_calendar_month(x) for x in timestamps]
#         self.days_of_the_week = [find_day_of_the_week(x) for x in timestamps]
#         calendar_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
#         				'Oct', 'Nov', 'Dec']
#         days_of_the_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
#         self.calendar_months_encoding = OneHotEncoder(self.calendar_months, calendar_months).encode()
#         self.days_of_the_week_encoding = OneHotEncoder(self.days_of_the_week, days_of_the_week).encode()
#         self.include_one_hot_encoding(self.calendar_months_encoding)
#         self.include_one_hot_encoding(self.days_of_the_week_encoding)
# 
# class SequenceLinearRegressionIncludingConvolutionalNetwork(SequenceLinearRegression):
#     def include_timestamps(self, timestamps):    
#         self.timestamps = timestamps
#         self.calendar_months = [find_calendar_month(x) for x in timestamps]
#         self.days_of_the_week = [find_day_of_the_week(x) for x in timestamps]
#         calendar_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
#         				'Oct', 'Nov', 'Dec']
#         days_of_the_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
#         self.calendar_months_encoding = OneHotEncoder(self.calendar_months, calendar_months).encode()
#         self.days_of_the_week_encoding = OneHotEncoder(self.days_of_the_week, days_of_the_week).encode()
#         self.include_one_hot_encoding(self.calendar_months_encoding)
#         self.include_one_hot_encoding(self.days_of_the_week_encoding)
# 
#     def create_convolutional_network(self):
#         self.cnn_input_layer = tf.placeholder(tf.float32, [len(self.dependent_variables), len(self.dependent_variables[0])])
#         self.cnn_input_layer_reshaped = tf.reshape(self.cnn_input_layer, [-1, len(self.dependent_variables[0]), 1])
#         self.cnn_conv1_filters = 5
#         self.cnn_conv1 = tf.layers.conv1d(self.cnn_input_layer_reshaped, filters = self.cnn_conv1_filters, kernel_size = 2, padding = "same", activation = tf.nn.relu)
#         self.cnn_last = self.cnn_conv1
#         number_of_cnn_outputs = 1
#         cnn_last_shape = tf.shape(self.cnn_last)
#         cnn_output_length = len(self.dependent_variables[0]) * self.cnn_conv1_filters   
#         self.cnn_last_flat = tf.reshape(self.cnn_last, [-1, cnn_output_length])
#         self.cnn_output = tf.layers.dense(self.cnn_last_flat, units = number_of_cnn_outputs, activation = tf.nn.sigmoid)
#         self.cnn_coefficients =  tf.Variable(tf.zeros([number_of_cnn_outputs, 1]))
#         self.cnn_to_add = tf.matmul(self.cnn_output, self.cnn_coefficients)
#         self.cnn_to_add = self.cnn_to_add - tf.reduce_mean(self.cnn_to_add)
#         self.y = self.y + self.cnn_to_add 
# 	
#     def train(self):      
#         datapoint_size = len(self.target)
#         batch_size = datapoint_size
#         self.create_linear_layers()
#         self.create_one_hot_layers()
#         self.create_convolutional_network()        
#         self.y_ = tf.placeholder(tf.float32, [None, 1])
#         self.cost = tf.reduce_mean(tf.square(self.y_ - self.y))
#         self.cost_sum = tf.summary.scalar("cost", self.cost)
#         self.train_step = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(self.cost)	
#         self.clip_op = tf.assign(self.W, tf.clip_by_value(self.W, 0, np.infty))
#         self.reduction_ops = {}
#         for i in xrange(len(self.one_hot_encodings)):
#             mean = tf.reduce_mean(self.one_hot_encodings_variables[str(i)])
#             self.reduction_ops[str(i)] = tf.assign(self.one_hot_encodings_variables[str(i)], self.one_hot_encodings_variables[str(i)] - mean)
#         sess = tf.Session()
#         init = tf.initialize_all_variables()
#         sess.run(init)
#         for i in xrange(self.steps):
#             print i
#             if datapoint_size == batch_size:
#                 batch_start_idx = 0
#             elif datapoint_size < batch_size:
#                 raise ValueError("datapoint_size: %d, must be greater than batch_size: %d" % (datapoint_size, batch_size))
#             else:
#                 batch_start_idx = (i * batch_size) % (datapoint_size - batch_size)
#             batch_end_idx = batch_start_idx + batch_size
#             batch_x = self.dependent_variables[batch_start_idx:batch_end_idx]
#             batch_y = self.target[batch_start_idx:batch_end_idx]
#             feed = {self.x: batch_x, self.y_: batch_y, self.cnn_input_layer: batch_x}
#             for j in xrange(len(self.one_hot_encodings)):
#                 to_feed = self.one_hot_encodings[j].encoding[batch_start_idx:batch_end_idx]
#                 feed[self.one_hot_encodings_placeholders[str(j)]] = to_feed
#             sess.run(self.train_step, feed_dict = feed)
#             if self.positive:
#                 sess.run(self.clip_op)
#             for j in xrange(len(self.one_hot_encodings)):
#             	sess.run(self.reduction_ops[str(j)])
#             print("After %d iterations:" % i)
#             print("W: %s" % sess.run(self.W))
#             if self.include_intercept:
#                 print("b: %f" % sess.run(self.b))
#             for j in xrange(len(self.one_hot_encodings)):
#                 print("one_hot_encodings_variable" + str(j) + " : %s" % sess.run(self.one_hot_encodings_variables[str(j)]))
#             print("Convolutional coefficient: %s" % sess.run(self.cnn_coefficients))










