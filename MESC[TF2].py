# Code Author:  Zhihao PENG,    City University of Hong Kong,        zhihapeng3-c@my.cityu.edu.hk
# Supervisor:   Junhui HOU,     City University of Hong Kong,        junhuhou@cityu.edu.hk
# [Remark]      The code is adapted from Pan (DSC-Net)
# Copyright Reserved!
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.contrib import layers
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import f1_score
from munkres import Munkres
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from datetime import datetime
import time
import math

tf.compat.v1.disable_eager_execution()

tic = time.time()
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

class ConvAE(object):
	def __init__(self, n_input, kernel_size, n_hidden, reg_const1 = 1.0, reg_const2 = 1.0, reg = None, batch_size = 256,\
		denoise = False, model_path = None, logs_path = './pretrain/logs'):	
		self.n_input 		= n_input
		self.n_hidden 		= n_hidden
		self.reg 			= reg
		self.model_path 	= model_path		
		self.kernel_size 	= kernel_size		
		self.iter 			= 0
		self.batch_size 	= batch_size
		self.x 				= tf.compat.v1.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 1])
		self.learning_rate 	= tf.compat.v1.placeholder(tf.float32, [])
		weights 			= self._initialize_weights()
		
		if denoise == False:
			x_input 		= self.x
			latent, shape 	= self.encoder(x_input, weights)
		else:
			x_input 		= tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
											   mean = 0,
											   stddev = 0.2,
											   dtype=tf.float32))
			latent,shape 	= self.encoder(x_input, weights)

		self.z_conv		 = tf.reshape(latent,[batch_size, -1])		
		self.z_ssc, Coef = self.selfexpressive_moduel(batch_size)	
		self.Coef 		 = Coef						
		# The key for the decoupling framework	
		self.x_r_ft	 	 = self.decoder(latent, weights, shape)		
		self.saver 	 	 = tf.compat.v1.train.Saver([v for v in tf.compat.v1.trainable_variables() if not (v.name.startswith("Coef"))]) 

		# Reconstruction loss 
		self.recon 	 	 =  tf.reduce_sum(tf.pow(tf.subtract(self.x_r_ft, self.x), 2.0))	
		# Maximum Entropy(ME) regularization loss ## The key for the affinity matrix
		self.reg_ssc 	 = 0.5*tf.reduce_sum( tf.multiply((self.Coef), tf.math.log( tf.compat.v1.clip_by_value(Coef, clip_value_min=1.0e-12, clip_value_max=1.0) )) )
		# Self-representation loss		
		self.cost_ssc 	 = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.z_conv,self.z_ssc), 2))

		tf.compat.v1.summary.scalar("self_expressive_loss", self.cost_ssc)
		tf.compat.v1.summary.scalar("coefficient_loss", self.reg_ssc)	
		tf.compat.v1.summary.scalar("reconstruction loss", self.recon)		
		self.loss_ssc = self.recon + reg_const1 * self.reg_ssc + reg_const2 * self.cost_ssc

		self.merged_summary_op 	= tf.compat.v1.summary.merge_all()		
		self.optimizer_ssc 		= tf.compat.v1.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss_ssc)
		self.init 				= tf.compat.v1.global_variables_initializer()
		self.sess 				= tf.compat.v1.InteractiveSession()
		self.summary_writer 	= tf.compat.v1.summary.FileWriter(logs_path, graph=tf.compat.v1.get_default_graph())
		self.sess.run(self.init)

	def _initialize_weights(self):
		all_weights 				= dict()
		n_layers 					= len(self.n_hidden)
		all_weights['enc_w0'] 		= tf.compat.v1.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]], initializer	=tf.compat.v1.keras.initializers.he_normal(),regularizer = self.reg)
		all_weights['enc_b0'] 		= tf.compat.v1.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32))
		iter_i 						= 1
		while iter_i < n_layers:
			enc_name_wi 				= 'enc_w' + str(iter_i)
			all_weights[enc_name_wi] 	= tf.compat.v1.get_variable(enc_name_wi, shape=[self.kernel_size[iter_i], self.kernel_size[iter_i], self.n_hidden[iter_i-1], self.n_hidden[iter_i]], initializer=tf.compat.v1.keras.initializers.he_normal(),regularizer = self.reg)
			enc_name_bi 				= 'enc_b' + str(iter_i)
			all_weights[enc_name_bi] 	= tf.compat.v1.Variable(tf.zeros([self.n_hidden[iter_i]], dtype = tf.float32))
			iter_i 						= iter_i + 1
		iter_i 						= 1
		while iter_i < n_layers:	
			dec_name_wi 				= 'dec_w' + str(iter_i - 1)
			all_weights[dec_name_wi] 	= tf.compat.v1.get_variable(dec_name_wi, shape=[self.kernel_size[n_layers-iter_i], self.kernel_size[n_layers-iter_i], self.n_hidden[n_layers-iter_i-1],self.n_hidden[n_layers-iter_i]], initializer=tf.compat.v1.keras.initializers.he_normal(),regularizer = self.reg)
			dec_name_bi 				= 'dec_b' + str(iter_i - 1)
			all_weights[dec_name_bi] 	= tf.compat.v1.Variable(tf.zeros([self.n_hidden[n_layers-iter_i-1]], dtype = tf.float32)) 
			iter_i 						= iter_i + 1
		dec_name_wi 				= 'dec_w' + str(iter_i - 1)
		all_weights[dec_name_wi] 	= tf.compat.v1.get_variable(dec_name_wi, shape=[self.kernel_size[0], self.kernel_size[0],1, self.n_hidden[0]], initializer=tf.compat.v1.keras.initializers.he_normal(),regularizer = self.reg)
		dec_name_bi 				= 'dec_b' + str(iter_i - 1)
		all_weights[dec_name_bi] 	= tf.compat.v1.Variable(tf.zeros([1], dtype = tf.float32))
		return all_weights	
	
	# Encoder
	def encoder(self,x, weights):
		shapes 		= []
		shapes.append(x.get_shape().as_list())
		layeri 		= tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1,2,2,1],padding='SAME'),weights['enc_b0'])
		layeri 		= tf.nn.relu(layeri)
		shapes.append(layeri.get_shape().as_list())
		n_layers 	= len(self.n_hidden)
		iter_i 		= 1
		while iter_i < n_layers:
			layeri 		= tf.nn.bias_add(tf.nn.conv2d(layeri, weights['enc_w' + str(iter_i)], strides=[1,2,2,1],padding='SAME'),weights['enc_b' + str(iter_i)])
			layeri 		= tf.nn.relu(layeri)
			shapes.append(layeri.get_shape().as_list())
			iter_i 		= iter_i + 1
		layer3 		= layeri
		return  layer3, shapes

	# Decoder
	def decoder(self,z, weights, shapes):
		n_layers 		= len(self.n_hidden)		
		layer3 			= z
		iter_i 			= 0
		while iter_i < n_layers:
			shape_de = shapes[n_layers - iter_i - 1] 		  
			layer3 	 = tf.add(tf.nn.conv2d_transpose(layer3, weights['dec_w' + str(iter_i)], tf.stack([tf.shape(self.x)[0],shape_de[1],shape_de[2],shape_de[3]]), strides=[1,2,2,1],padding='SAME'), weights['dec_b' + str(iter_i)])
			layer3 	 = tf.nn.relu(layer3)
			iter_i 	 = iter_i + 1
		return layer3

	def selfexpressive_moduel(self,batch_size):
		Coef 	= tf.compat.v1.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size],tf.float32), name = 'Coef')
		z_ssc 	= tf.matmul(Coef, self.z_conv)
		return z_ssc, Coef

	def finetune_fit(self, X, lr):
		C, l_cost, l1_cost, l2_cost, summary, _ = self.sess.run((self.Coef, self.loss_ssc, self.reg_ssc, self.cost_ssc, self.merged_summary_op, self.optimizer_ssc), feed_dict = {self.x: X, self.learning_rate: lr})
		self.summary_writer.add_summary(summary, self.iter)
		self.iter 								= self.iter + 1
		return C, l_cost, l1_cost, l2_cost 
	
	def initlization(self):
		self.sess.run(self.init)	

	# For the close of interactive session
	def runclose(self):
		self.sess.close()
		print("InteractiveSession.close()")
		
	def restore(self):
		self.saver.restore(self.sess, self.model_path)
		print ("model restored")

# L1: Groundtruth labels; L2: Clustering labels;
def best_map(L1,L2):
	Label1 	= np.unique(L1)
	nClass1 = len(Label1)
	Label2 	= np.unique(L2)
	nClass2 = len(Label2)
	nClass 	= np.maximum(nClass1,nClass2)
	G 		= np.zeros((nClass,nClass))
	for i in range(nClass1):
		ind_cla1 = L1 == Label1[i]
		ind_cla1 = ind_cla1.astype(float)
		for j in range(nClass2):
			ind_cla2 = L2 == Label2[j]
			ind_cla2 = ind_cla2.astype(float)
			G[i,j] 	 = np.sum(ind_cla2 * ind_cla1)
	m 		= Munkres()
	index 	= m.compute(-G.T)
	index 	= np.array(index)
	c 		= index[:,1]
	newL2 	= np.zeros(L2.shape)
	for i in range(nClass2):
		newL2[L2 == Label2[i]] = Label1[c[i]]
	return newL2

def thrC(C,ro):
	if ro < 1:
		N 	= C.shape[1]
		Cp 	= np.zeros((N,N))
		S 	= np.abs(np.sort(-np.abs(C),axis=0))
		Ind = np.argsort(-np.abs(C),axis=0)
		for i in range(N):
			cL1 	= np.sum(S[:,i]).astype(float)
			stop 	= False
			csum 	= 0
			t 		= 0
			while(stop == False):
				csum 	= csum + S[t,i]
				if csum > ro*cL1:
					stop = True
					Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
				t 		= t + 1
	else:
		Cp = C
	return Cp

# C: coefficient matrix; K: number of clusters; d: dimension of each subspace;
def post_proC(C, K, d, alpha):
	n 		= C.shape[0]
	C	 	= C - np.diag(np.diag(C)) + np.eye(n,n)
	r 		= d*K + 1	
	U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
	U 		= U[:,::-1] 
	S 		= np.sqrt(S[::-1])
	S 		= np.diag(S)
	U 		= U.dot(S)
	U 		= normalize(U, norm='l2', axis = 1)  
	Z 		= U.dot(U.T)
	Z 		= Z * (Z>0)
	L 		= np.abs(Z ** alpha)
	L 		= L/L.max()
	L 		= 0.5 * (L + L.T)	
	spectral= cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
	spectral.fit(L)
	grp 	= spectral.fit_predict(L) + 1 
	return grp, L

def err_rate(gt_s, s):
	c_x 		= best_map(gt_s,s)
	err_x 		= np.sum(gt_s[:] != c_x[:])
	missrate 	= err_x.astype(float) / (gt_s.shape[0])
	return missrate  

def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

# COIL40
data 	= sio.loadmat('./Data/COIL100.mat')
Img 	= data['fea'][0:40*72]
Label 	= data['gnd'][0:40*72]
Img 	= np.reshape(Img,(Img.shape[0],32,32,1))

n_input 	= [32,32]
kernel_size = [3]
n_hidden 	= [20]
batch_size 	= 40*72

model_path 	= './pretrain-model-COIL40/model-32x32-coil40.ckpt'
ft_path 	= './pretrain-model-COIL40/model-32x32-coil40.ckpt'
logs_path 	= './pretrain-model-COIL40/ft/logs' + TIMESTAMP

num_class 		= 40
num_sa 			= 72
batch_size_test = num_sa * num_class
alpha 			= 0.03
reg1 			= 1.0
reg2 			= 1e-1
learning_rate 	= 1.0e-4
iter_ft 		= 0
ft_times 		= 127 # empirical_r = 0.89 
display_step 	= ft_times # - 1

acc_	= []
nmi_	= []
pur_	= []
ari_	= []
f1score_= []
loss_	= []

CAE = ConvAE(n_input = n_input, n_hidden = n_hidden, reg_const1 = reg1, reg_const2 = reg2, kernel_size = kernel_size, \
			batch_size = batch_size_test, model_path = model_path, logs_path= logs_path)

for i in range(0,1):
	coil40_all_subjs	= np.array(Img[i*num_sa:(i+num_class)*num_sa,:])
	coil40_all_subjs	= coil40_all_subjs.astype(float)	
	label_all_subjs		= np.array(Label[i*num_sa:(i+num_class)*num_sa])    
	label_all_subjs 	= label_all_subjs - label_all_subjs.min() + 1    
	label_all_subjs 	= np.squeeze(label_all_subjs)  	

	CAE.initlization()
	CAE.restore()

	for iter_ft  in range(ft_times):
		iter_ft = iter_ft+1
		C,l_cost,l1_cost,l2_cost = CAE.finetune_fit(coil40_all_subjs,learning_rate)
		if (iter_ft % display_step == 0):
			C 				= thrC(C,alpha)
			y_x, CKSym_x 	= post_proC(C, num_class, 12, 8)	
			loss_.append(l_cost)
			# ACC
			missrate_x 		= err_rate(label_all_subjs,y_x)			
			acc 			= 1 - missrate_x
			print ("acc:",acc,"iter_ft:",iter_ft) 
			acc_.append(acc)

			# NMI 
			nmi_x = normalized_mutual_info_score(label_all_subjs, y_x)
			nmi_.append(nmi_x)

			# PUR 
			pur_x = purity_score(label_all_subjs, y_x)
			pur_.append(pur_x)

			# ARI
			ari_x = adjusted_rand_score(label_all_subjs, y_x)
			ari_.append(ari_x)

			# f1score
			f1score_x = f1_score((label_all_subjs), (y_x), average='micro')
			f1score_.append(f1score_x)

	CAE.runclose()
	tf.compat.v1.reset_default_graph()

# ACC
acc_ 		= np.array(acc_)
acc_mean 	= np.mean(acc_)
print("acc_mean:",acc_mean)
print(acc_) 

# NMI 
nmi_ 		= np.array(nmi_)
nmi_mean 	= np.mean(nmi_)
print("nmi_mean:",nmi_mean)
print(nmi_) 

# PUR 
pur_ 		= np.array(pur_)
pur_mean 	= np.mean(pur_)
print("pur_mean:",pur_mean)
print(pur_) 

# ARI
ari_ 		= np.array(ari_)
ari_mean 	= np.mean(ari_)
print("ari_mean:",ari_mean) 
print(ari_) 

# flscore
f1score_ 	= np.array(f1score_)
f1_mean 	= np.mean(f1score_)
print("f1_mean:",f1_mean)
print(f1score_) 

# Record the running time (s)
toc = time.time()
print("Time:", (toc - tic))