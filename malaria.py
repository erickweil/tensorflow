
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tkinter import Tk, Label, Button, Canvas
from PIL import Image, ImageTk, ImageDraw

import numpy as np
import argparse
import sys
import os
import random

# Add the ptdraft folder path to the sys.path list
sys.path.append('D:/tensorflow')
import networkIO

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

# faces v2
INPUT_WIDTH = 64
INPUT_HEIGHT = 64
INPUT_WIDTH_d2 = 32
INPUT_HEIGHT_d2 = 32
INPUT_WIDTH_d4 = 16
INPUT_HEIGHT_d4 = 16
INPUT_DEPTH = 3

IMSIZE = INPUT_WIDTH*INPUT_HEIGHT*INPUT_DEPTH



batch_index = 0
mb_size = 30
def carregarfaces_v2(pasta):
	img_files = os.listdir(pasta)
	file_count = min(1000,len(img_files))
	
	loaded_dataset = np.zeros((file_count,INPUT_WIDTH*INPUT_HEIGHT*INPUT_DEPTH), dtype=np.float32)
	i = 0
	for img_file in img_files:
		image = Image.open(pasta+"/"+img_file)
		image = image.resize((INPUT_WIDTH,INPUT_HEIGHT), Image.ANTIALIAS)
		#pix_im = image.convert('L') # preto e branco
		pix_im = image.convert('RGB')
		index = 0
		for r,g,b in pix_im.getdata():
			loaded_dataset[i][index+0] = float(r)/255.0
			loaded_dataset[i][index+1] = float(g)/255.0
			loaded_dataset[i][index+2] = float(b)/255.0
			index += 3
		#for r in pix_im.getdata():
		#	loaded_dataset[i][index] = float(r)/255.0
		#	index += 1
		i += 1
		if i >= file_count:
			break
		
	return loaded_dataset
def next_batch(n,parazited_dataset,health_dataset):
	global batch_index
	batch = np.zeros((n,INPUT_WIDTH*INPUT_HEIGHT*INPUT_DEPTH), dtype=np.float32)
	label = np.zeros((n,2), dtype=np.float32)
	for i in range(n):
		batch_index += 1
		if(batch_index >= len(parazited_dataset)):
			batch_index = 0
		if(random.random() > 0.5):
			# PARAZITED
			batch[i] = parazited_dataset[batch_index]
			label[i] = [0,1]
		else:
			# HEALTH
			batch[i] = health_dataset[batch_index]
			label[i] = [1,0]
	return (batch,label)
	
def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
					strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
					strides=[1, 2, 2, 1], padding='SAME')

net_var_names = {}
net_var_objs = {}
def load_network(file):
	global net_var_names
	if os.path.isfile(file):
		net_var_names = networkIO.load(file)
def save_network(file,sess):
	for key, value in net_var_objs.items():
		var_value = sess.run(value)
		net_var_names[key] = var_value
	networkIO.save(file,"MNIST,IN 784,OUT 10, CPCPFDF, relu",net_var_names)

def weight_variable(name,shape):
	"""weight_variable generates a weight variable of a given shape."""
	#initial = tf.truncated_normal(shape, stddev=0.1)
	#return tf.Variable(initial)
	if name in net_var_names:
		v = tf.Variable(net_var_names[name],name=name)
	else:
		v = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
	net_var_objs[name] = v
	return v

def bias_variable(name,shape):
	"""bias_variable generates a bias variable of a given shape."""
	if name in net_var_names:
		v = tf.Variable(net_var_names[name],name=name)
	else:
		initial = tf.constant(0.1, shape=shape)
		v = tf.Variable(initial,name=name)
	net_var_objs[name] = v
	return v

def init_network():

	#conf
	neurons_input = IMSIZE
	neurons_hidden1 = 128
	neurons_hidden2 = 128
	neurons_output = 2
	
	#type
	net_conv = True
	net_multi = False

	# Create the model
	net_input = tf.placeholder(tf.float32, [None, neurons_input])

	if net_multi:
		# primeira camada
		#W1 = tf.Variable(tf.zeros([neurons_input, neurons_hidden1]))
		W1 = tf.get_variable("W1", shape=[neurons_input, neurons_hidden1],initializer=tf.contrib.layers.xavier_initializer())
		b1 = tf.Variable(tf.zeros([neurons_hidden1]))
		y1 = tf.nn.relu(tf.matmul(net_input, W1) + b1)

		#segunda camada HIDDEN 1
		W2 = tf.get_variable("W2", shape=[neurons_hidden1, neurons_hidden2],initializer=tf.contrib.layers.xavier_initializer())
		b2 = tf.Variable(tf.zeros([neurons_hidden2]))
		y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)

		#terceira camada
		W3 = tf.get_variable("W3", shape=[neurons_hidden2, neurons_output],initializer=tf.contrib.layers.xavier_initializer())
		b3 = tf.Variable(tf.zeros([neurons_output]))
		y3 = tf.matmul(y2, W3) + b3
		
		net_output = y3
	
	if net_conv:
		image_input = tf.reshape(net_input, [-1, INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH])
		
		
		load_network("digits.neu")
		# weight and biases
		# first convolution
		W_conv1 = weight_variable("w_conv1",[5, 5, INPUT_DEPTH, 12])
		b_conv1 = bias_variable("b_conv1",[12])
		
		# second convolution
		W_conv2 = weight_variable("w_conv2",[5, 5, 12, 24])
		b_conv2 = bias_variable("b_conv2",[24])
		
		# fully connected 1
		W_fc1 = weight_variable("w_fc1",[INPUT_HEIGHT_d4 * INPUT_HEIGHT_d4 * 24, 64])
		b_fc1 = bias_variable("b_fc1",[64])
		
		# fully connected 2
		W_fc2 = weight_variable("w_fc2",[64, 2])
		b_fc2 = bias_variable("b_fc2",[2])
		
		# network
		# first convolution
		h_conv1 = tf.nn.relu(conv2d(image_input, W_conv1) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)
		
		# second convolution
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = max_pool_2x2(h_conv2)
		
		# fully connected 1
		# 28x28 -> 14x14 -> 7x7, imgs 7x7, 8 of them
		h_pool2_flat = tf.reshape(h_pool2, [-1, INPUT_HEIGHT_d4 * INPUT_HEIGHT_d4 * 24])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
		
		# prevent overfitting 
		# drop some neurons, should be 1.0 when testing accuracy
		#keep_prob = tf.placeholder(tf.float32)
		keep_prob = tf.constant(0.5)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		# fully connected 2
		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	
		net_output = y_conv
	# Define loss and optimizer
	net_ideal = tf.placeholder(tf.float32, [None, 2])

	# The raw formulation of cross-entropy,
	#
	#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
	#								 reduction_indices=[1]))
	#
	# can be numerically unstable.
	#
	# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
	# outputs of 'y', and then average across the batch.
	net_loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=net_ideal, logits=net_output))
	#net_train = tf.train.GradientDescentOptimizer(0.5).minimize(net_loss)
	net_train = tf.train.AdamOptimizer(0.5e-3).minimize(net_loss)

	return (net_input,net_output,net_ideal,net_loss,net_train)

def start():
	global mnist, net_input,net_output,net_ideal,net_loss,net_train,sess, parazited_dataset, health_dataset
	# Import data
	#mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
	parazited_dataset = carregarfaces_v2("D:/tensorflow/datasets/cell_images/Parasitized")
	health_dataset = carregarfaces_v2("D:/tensorflow/datasets/cell_images/Uninfected")

	net_input,net_output,net_ideal,net_loss,net_train = init_network()

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
# Train
def treinar(i):
	global lastExample, lastY
	#for i in range(50000):
	#batch_xs, batch_ys = mnist.train.next_batch(100)
	batch_xs,batch_ys = next_batch(mb_size,parazited_dataset,health_dataset)
	
	lastExample = batch_xs[0]
	lastY = batch_ys[0]
	
	train_result, loss_result = sess.run([net_train,net_loss], feed_dict={net_input: batch_xs, net_ideal: batch_ys})
	if(i % 500 == 0):
		print(loss_result)
		save_network("malaria.neu",sess)
		
		
def testar(data):
	print("################################")
	print("####	   TESTANDO		 ####")
	print("################################")
	#Test trained model
	#correct_prediction = tf.equal(tf.argmax(net_output, 1), tf.argmax(net_ideal, 1))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	#print(sess.run(accuracy, feed_dict={net_input: mnist.test.images,
	##	net_ideal: mnist.test.labels}))
		

	possibilidades = sess.run(net_output, feed_dict={net_input: [data]})[0]
	escolhido = 0
	melhor = possibilidades[0]
	for i in range(2):
		if melhor < possibilidades[i]:
			escolhido = i
			melhor = possibilidades[i]
	if escolhido == 0:
		print("Saudável! w:",melhor," array:",possibilidades)
	else:
		print("Infectada! w:",melhor," array:",possibilidades)
	
def print_image(im,width,height,scale):
	#print("len(im)   :",len(im))
	#print("len(im[0] :)",len(im[0]))
	data = np.zeros((height,width, 3), dtype=np.uint8)
	for i in range(width*height):
		#w = im[i]
		
		r = max(min(im[i*3+0],1.0),0.0)
		g = max(min(im[i*3+1],1.0),0.0)
		b = max(min(im[i*3+2],1.0),0.0)
		
		r = int(r*255.0)
		g = int(g*255.0)
		b = int(b*255.0)
		
		x = int(i % width)
		y = int(i / width)
		data[y,x] = [r,g,b]

	img = Image.fromarray(data, 'RGB')
	img = img.resize((width*scale,height*scale), Image.NEAREST)
	#img.save('my.png')
	#img.show()
	#var = input("Please enter something: ")
	return img
	
class MyFirstGUI:
	def __init__(self, master):
		self.master = master
		master.title("A simple GUI")
		
		self.scale = 8
		self.imwidth = 64*self.scale
		
		self.canvas = Label(master)
		self.canvas.pack()
		self.canvas.bind( "<Button-1>", self.startpaint )
		self.canvas.bind( "<B1-Motion>", self.paint )
		self.canvas.bind( "<ButtonRelease-1>", self.stoppaint )
		
		self.greet_button = Button(master, text="Testar", command=self.testar)
		self.greet_button.pack()
		self.Limpar_button = Button(master, text="Limpar", command=self.limpar)
		self.Limpar_button.pack()
		
		self.Exemplo_button = Button(master, text="Exemplo", command=self.exemplo)
		self.Exemplo_button.pack()

		self.close_button = Button(master, text="Close", command=master.quit)
		self.close_button.pack()
		self.lastx = 0
		self.lasty = 0
		self.i = 0
		
		self.image = Image.new("RGB", (self.imwidth, self.imwidth), (0,0,0))
		self.imagecanvas = ImageDraw.Draw(self.image)
		
		self.photo = ImageTk.PhotoImage(self.image)
		self.canvas.configure(image=self.photo)
		
		start()
		self.repetir()
		
	def repetir(self):
		treinar(self.i)
		self.i += 1
		self.master.after(1, self.repetir)
		
	def testar(self):
		impixels = self.image.resize((INPUT_WIDTH,INPUT_HEIGHT), Image.ANTIALIAS).load()
		data = np.zeros((IMSIZE), dtype=np.float32)
		index = 0
		for x in range(INPUT_WIDTH):
			for y in range(INPUT_HEIGHT):
				cpixel = impixels[y, x][0]
				data[index] = cpixel
				index += 1
		#print("OK")
		testar(data)
		#self.image = Image.open("teste.jpg")
		#self.photo = ImageTk.PhotoImage(self.image)
		#self.label.configure(image=self.photo)
	
	def startpaint( self,event ):
		self.lastx = event.x
		self.lasty = event.y
	
	def stoppaint( self,event ):
		self.lastx = 0
		self.lasty = 0
		
	def limpar( self ):
		self.image = Image.new("RGB", (self.imwidth, self.imwidth), (0,0,0))
		self.imagecanvas = ImageDraw.Draw(self.image)
		
		self.photo = ImageTk.PhotoImage(self.image)
		self.canvas.configure(image=self.photo)
		
	def exemplo( self ):
		print("Label:",lastY)
		self.image = print_image(lastExample,INPUT_WIDTH,INPUT_HEIGHT,self.scale)
		self.imagecanvas = ImageDraw.Draw(self.image)
		
		self.photo = ImageTk.PhotoImage(self.image)
		self.canvas.configure(image=self.photo)
	
	def paint( self,event ):	
		#self.canvas.create_line(self.lastx, self.lasty,				 # origin of canvas
		#			  event.x , event.y, # coordinates of left upper corner of the box[0]
		#			  fill="#FFFFFF", 
		#			  width=self.scale*2)
		self.lastx = event.x
		self.lasty = event.y
		x1, y1 = ( event.x - self.scale ), ( event.y - self.scale )
		x2, y2 = ( event.x + self.scale ), ( event.y + self.scale )
		self.imagecanvas.ellipse( [(x1, y1), (x2, y2)], outline = (255,255,255), fill = (255,255,255) )
		t = self.image.resize((28,28), Image.ANTIALIAS)
		t = t.resize((self.imwidth,self.imwidth), Image.NEAREST)
		self.photo = ImageTk.PhotoImage(t)
		self.canvas.configure(image=self.photo)

def main(_):
	root = Tk()
	my_gui = MyFirstGUI(root)
	root.mainloop()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
					  help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)