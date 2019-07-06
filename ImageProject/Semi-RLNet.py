#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:14:00 2017

@author: ubuntu-507
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy
import tensorflow as tf
import input_data_paviau
import csv
from tqdm import tqdm
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 103
NUM_CHANNELS = 1
NUM_LABELS = 9
SEED = 66478  # Set to None for random seed.

FLAGS = None


num_examples = 8551
num_epochs = 100
num_labeled = 0.1

starter_learning_rate = 0.0001

decay_after = 15  # epoch after which to begin learning rate decay

batch_size = 100
num_iter = int((num_examples/batch_size) * num_epochs)  # number of loop iterations


inputs = tf.placeholder(tf.float32,shape=[None,103])
re_inputs=tf.reshape(inputs,[-1,1,103,1])  
outputs = tf.placeholder(tf.float32)


layer_sizes=[batch_size,16,16,32,64]

conv_layer_sizes=[16,16,16,16,16,32,32,32,32,64,64,64,64,64,64,64,64,32,32,32,32,16,16,16,16,16,16]

noise_std = 0.1  


denoising_cost = [0.001, 0.0001, 0.0001, 0.0001,0.0001]

join = lambda l, u: tf.concat([l, u], 0)  #堆叠到一起
labeled = lambda x: tf.slice(x, [0,0,0,0], [batch_size, -1,-1,-1]) if x is not None else x
unlabeled = lambda x: tf.slice(x, [batch_size, 0,0,0], [batch_size, -1,-1,-1]) if x is not None else x
split_lu = lambda x: (labeled(x), unlabeled(x))

join_fc = lambda l, u: tf.concat([l, u], 0)
labeled_fc = lambda x: tf.slice(x, [0, 0], [batch_size, -1]) if x is not None else x
unlabeled_fc = lambda x: tf.slice(x, [batch_size, 0], [-1, -1]) if x is not None else x
split_lu_fc = lambda x: (labeled_fc(x), unlabeled_fc(x))

training = tf.placeholder(tf.bool)
  
def weight_variable(shape):  
    initial = tf.truncated_normal(shape,stddev=0.1,dtype=tf.float32)   
    return tf.Variable(initial)  
def bias_variable(shape):  
    initial = tf.zeros(shape, dtype=tf.float32)  
    return tf.Variable(initial)
def fc_bias_variable(shape):  
    initial = tf.constant(0.1,shape=shape,dtype=tf.float32)  
    return tf.Variable(initial)  
def conv2d(x,W):  
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID') 
def conv2d_p(x,W):  
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME') 
def conv2d_s2(x,W):  
    return tf.nn.conv2d(x,W,strides=[1,1,2,1],padding='VALID')  
def max_pool_2x2(x):  
    return tf.nn.max_pool(x,ksize=[1, 1, 3, 1],strides=[1, 1, 2, 1],padding='VALID')
def de_conv2d(x,W,output):  
    return tf.nn.conv2d_transpose(x,W,output_shape=output,strides=[1,1,1,1],padding='VALID') 
def de_conv2d_p(x,W,output):  
    return tf.nn.conv2d_transpose(x,W,output_shape=output,strides=[1,1,1,1],padding='SAME') 
def de_conv2d_s2(x,W,output):  
    return tf.nn.conv2d_transpose(x,W,output_shape=output,strides=[1,1,2,1],padding='VALID') 
    return tf.nn.conv2d_transpose(x,W,output_shape=output,strides=[1,1,2,1],padding='VALID') 



conv1_weights=weight_variable([1, 7, NUM_CHANNELS, 16])
conv1_biases=bias_variable([16])

conv2_weights=weight_variable([1, 1, 16, 16])
conv2_biases=bias_variable([16])

uint0_conv1_weights=weight_variable([1, 1, 16, 16])
uint0_conv1_biases=bias_variable([16])
uint0_conv2_weights=weight_variable([1, 3, 16, 16])
uint0_conv2_biases=bias_variable([16])
uint0_conv3_weights=weight_variable([1, 1, 16, 16])
uint0_conv3_biases=bias_variable([16])

conv3_weights=weight_variable([1, 1, 16, 32])
conv3_biases=bias_variable([32])

uint1_conv1_weights=weight_variable([1, 1, 32, 32])
uint1_conv1_biases=bias_variable([32])
uint1_conv2_weights=weight_variable([1, 3, 32, 32])
uint1_conv2_biases=bias_variable([32])
uint1_conv3_weights=weight_variable([1, 1, 32, 32])
uint1_conv3_biases=bias_variable([32])


conv4_weights=weight_variable([1, 1, 32, 64])
conv4_biases=bias_variable([64])

uint2_conv1_weights=weight_variable([1, 1, 64, 64])
uint2_conv1_biases=bias_variable([64])
uint2_conv2_weights=weight_variable([1, 3, 64, 64])
uint2_conv2_biases=bias_variable([64])
uint2_conv3_weights=weight_variable([1, 1, 64, 64])
uint2_conv3_biases=bias_variable([64])


fc1_weights = weight_variable([1*25*64, 96])
fc1_biases = fc_bias_variable([96])
fc2_weights = weight_variable([96, NUM_LABELS])
fc2_biases =fc_bias_variable([NUM_LABELS])




defc2_weights = weight_variable([NUM_LABELS,96])
defc2_biases = fc_bias_variable([96])
defc1_weights = weight_variable([96,1*25*64])
defc1_biases = fc_bias_variable([1*25*64])


running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in conv_layer_sizes[0:]]
running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in conv_layer_sizes[0:]]

ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
bn_assigns = []  # this list stores the updates to be made to average mean and variance


def update_batch_normalization(batch,axis, l):
    "batch normalize + update average mean and variance of layer l"
    mean, var = tf.nn.moments(batch, axis)
    assign_mean = running_mean[l].assign(mean)
    assign_var = running_var[l].assign(var)
    bn_assigns.append(ewma.apply([running_mean[l], running_var[l]]))
    with tf.control_dependencies([assign_mean, assign_var]):
        return (batch - mean) / tf.sqrt(var + 1e-10)
def batch_normalization(batch, axis,l,mean=None, var=None):
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch,axis)
    return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))
def bn(z_pre,training,fc,noise_std,l):
     if fc==0:
          z_pre_u,z_pre_l=split_lu(z_pre)
     else:
          z_pre_u,z_pre_l=split_lu_fc(z_pre)
     z_pre_shape = z_pre.get_shape()
     axis = list(range(len(z_pre_shape) - 1))
     m, v = tf.nn.moments(z_pre_u, axis)
     def training_batch_norm():
        if noise_std > 0:
            # Corrupted encoder
            z = join(batch_normalization(z_pre_l,axis,l), batch_normalization(z_pre_u,axis,l,m, v))
            if l==0:
                z += tf.random_normal(tf.shape(z_pre)) * noise_std

#            elif l==1:
#                z += tf.random_normal(tf.shape(z_pre)) *noise_std
#            elif l==5:
#                z += tf.random_normal(tf.shape(z_pre)) *noise_std
#            elif l==9:
#                z += tf.random_normal(tf.shape(z_pre)) *noise_std
        else:
            # Clean encoder
            z = join(batch_normalization(z_pre_l,axis,l), batch_normalization(z_pre_u,axis,l,m, v))
        return z
     def eval_batch_norm():
            # Evaluation batch normalization
        mean = ewma.average(running_mean[l])
        var = ewma.average(running_var[l])
        z = batch_normalization(z_pre,axis,l,mean, var)
        return z 
     z = tf.cond(training, training_batch_norm, eval_batch_norm)
     return z
def encoder(re_inputs,noise_std):
    re_inputs=re_inputs+tf.random_normal(tf.shape(re_inputs)) * noise_std
    d = {}
    d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}} 
    with tf.name_scope('conv1') as scope:
        kernel = conv1_weights
        conv = conv2d_s2(re_inputs, kernel)
        conv1 = tf.nn.bias_add(conv, conv1_biases)
    with tf.name_scope('bn1') as scope:
        b_norm=bn(conv1, training,0,noise_std,0)
        bn1 = tf.nn.relu(b_norm,name=scope)
    #adapt layer0    
    with tf.name_scope('conv2') as scope:
        kernel = conv2_weights
        conv = conv2d(bn1, kernel)
        conv2 = tf.nn.bias_add(conv, conv2_biases)
    with tf.name_scope('bn2') as scope:
        bn2=bn(conv2, training,0,noise_std,1)
    #uint0
    with tf.name_scope('uint0_conv1') as scope:
        kernel = uint0_conv1_weights
        conv = conv2d(bn2, kernel)
        uint0_conv1 = tf.nn.bias_add(conv, uint0_conv1_biases)
    with tf.name_scope('uint0_bn1') as scope:
        b_norm=bn(uint0_conv1, training,0,noise_std,2)
        uint0_bn1 = tf.nn.relu(b_norm,name=scope)
    with tf.name_scope('uint0_conv2') as scope:
        kernel = uint0_conv2_weights
        conv = conv2d_p(uint0_bn1, kernel)
        uint0_conv2 = tf.nn.bias_add(conv, uint0_conv2_biases)
    with tf.name_scope('uint0_bn2') as scope:
        b_norm=bn(uint0_conv2, training,0,noise_std,3)
        uint0_bn2 = tf.nn.relu(b_norm,name=scope)
    with tf.name_scope('uint0_conv3') as scope:
        kernel = uint0_conv3_weights
        conv = conv2d(uint0_bn2, kernel)
        uint0_conv3 = tf.nn.bias_add(conv, uint0_conv3_biases)
    with tf.name_scope('uint0_bn3') as scope:
        uint0_bn3=bn(uint0_conv3, training,0,noise_std,4)
#        uint0_bn3 = tf.nn.elu(b_norm,name=scope)
    with tf.name_scope('residual0_sum') as scope:
        residual0_sum=uint0_bn3+bn2
        residual0_sum = tf.nn.relu(residual0_sum,name=scope)

    #adapt layer1
    with tf.name_scope('conv3') as scope:
        kernel = conv3_weights
        conv = conv2d_s2(residual0_sum, kernel)
        conv3 = tf.nn.bias_add(conv, conv3_biases)
    with tf.name_scope('bn3') as scope:
        bn3=bn(conv3, training,0,noise_std,5)

    #uint1  
    with tf.name_scope('uint1_conv1') as scope:
        kernel = uint1_conv1_weights
        conv = conv2d(bn3, kernel)
        uint1_conv1 = tf.nn.bias_add(conv, uint1_conv1_biases)
    with tf.name_scope('uint1_bn1') as scope:
        b_norm=bn(uint1_conv1, training,0,noise_std,6)
        uint1_bn1 = tf.nn.relu(b_norm,name=scope)
    with tf.name_scope('uint1_conv2') as scope:
        kernel = uint1_conv2_weights
        conv = conv2d_p(uint1_bn1, kernel)
        uint1_conv2 = tf.nn.bias_add(conv, uint1_conv2_biases)
    with tf.name_scope('uint1_bn2') as scope:
        b_norm=bn(uint1_conv2, training,0,noise_std,7)
        uint1_bn2 = tf.nn.relu(b_norm,name=scope)
    with tf.name_scope('uint1_conv3') as scope:
        kernel = uint1_conv3_weights
        conv = conv2d(uint1_bn2, kernel)
        uint1_conv3 = tf.nn.bias_add(conv, uint1_conv3_biases)
    with tf.name_scope('uint1_bn3') as scope:
        uint1_bn3=bn(uint1_conv3, training,0,noise_std,8)
#        uint1_bn3 = tf.nn.elu(b_norm,name=scope)
    with tf.name_scope('residual1_sum') as scope:
        residual1_sum=uint1_bn3+bn3
        residual1_sum = tf.nn.relu(residual1_sum,name=scope)
    #adapt layer2
    with tf.name_scope('conv4') as scope:
        kernel = conv4_weights
        conv = conv2d(residual1_sum, kernel)
        conv4 = tf.nn.bias_add(conv, conv4_biases)
    with tf.name_scope('bn4') as scope:
        bn4=bn(conv4, training,0,noise_std,9)
    #uint2  
    with tf.name_scope('uint2_conv1') as scope:
        kernel = uint2_conv1_weights
        conv = conv2d(bn4, kernel)
        uint2_conv1 = tf.nn.bias_add(conv, uint2_conv1_biases)
    with tf.name_scope('uint2_bn1') as scope:
        b_norm=bn(uint2_conv1, training,0,noise_std,10)
        uint2_bn1 = tf.nn.relu(b_norm,name=scope)
    with tf.name_scope('uint2_conv2') as scope:
        kernel = uint2_conv2_weights
        conv = conv2d_p(uint2_bn1, kernel)
        uint2_conv2 = tf.nn.bias_add(conv, uint2_conv2_biases)
    with tf.name_scope('uint2_bn2') as scope:
        b_norm=bn(uint2_conv2, training,0,noise_std,11)
        uint2_bn2 = tf.nn.relu(b_norm,name=scope)
    with tf.name_scope('uint2_conv3') as scope:
        kernel = uint2_conv3_weights
        conv = conv2d(uint2_bn2, kernel)
        uint2_conv3 = tf.nn.bias_add(conv, uint2_conv3_biases)
    with tf.name_scope('uint2_bn3') as scope:
        uint2_bn3=bn(uint2_conv3, training,0,noise_std,12)
#        uint2_bn3 = tf.nn.elu(b_norm,name=scope)
    with tf.name_scope('residual2_sum') as scope:
        residual2_sum=uint2_bn3+bn4
        residual2_sum = tf.nn.relu(residual2_sum,name=scope)
#    with tf.name_scope('avg_pool') as scope:
#        avg_pool = avg_pool_2x2(residual2_sum)
    pool_shape = tf.shape(residual2_sum)
    reshape = tf.reshape(residual2_sum,[-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])
    fc1 = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    fc2=tf.matmul(fc1, fc2_weights) + fc2_biases
    last_prediction= fc2
    d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(re_inputs)
    d['labeled']['z'][1], d['unlabeled']['z'][1] = split_lu(bn1)
    d['labeled']['z'][2], d['unlabeled']['z'][2] = split_lu(residual0_sum)
    d['labeled']['z'][3], d['unlabeled']['z'][3] = split_lu(residual1_sum)
    d['labeled']['z'][4], d['unlabeled']['z'][4] = split_lu(residual2_sum)
    return tf.nn.softmax(last_prediction),d,re_inputs,conv1,conv2,conv3,conv4,fc1,fc2

def decoder(re_data,cor_conv1,cor_conv2,cor_conv3,cor_conv4,cor_fc1,cor_fc2):
    d_conv = {}
    d_conv['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    d_conv['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    with tf.name_scope('defc1') as scope:
        kernel = defc1_weights 
        biases = defc1_biases    
        defc1=tf.matmul(cor_fc1, kernel) + biases
        input_shape=cor_conv4.get_shape().as_list()
        defc1=tf.reshape(defc1, [-1, input_shape[1] ,input_shape[2] , input_shape[3]])
    with tf.name_scope('de_uint2_bn3') as scope:
        de_uint2_bn3=bn(defc1, training,0,noise_std,13) 
#        de_uint2_bn3_r= tf.nn.relu(de_uint2_bn3,name=scope)     
    with tf.name_scope('de_uint2_conv2') as scope:  
        kernel = weight_variable([1, 1, 64, 64])
        output_shape=tf.shape(cor_conv4)
        deconv = de_conv2d(de_uint2_bn3, kernel,output_shape)
        biases = bias_variable([64])
        de_uint2_conv2 = tf.nn.bias_add(deconv, biases)
    with tf.name_scope('de_uint2_bn2') as scope:
        b_norm=bn(de_uint2_conv2, training,0,noise_std,14)
        de_uint2_bn2= tf.nn.relu(b_norm,name=scope)   
    with tf.name_scope('de_uint2_conv1') as scope:  
        kernel = weight_variable([1, 3, 64, 64])
        output_shape=tf.shape(cor_conv4)
        deconv = de_conv2d_p(de_uint2_bn2, kernel,output_shape)
        biases = bias_variable([64])
        de_uint2_conv1 = tf.nn.bias_add(deconv, biases)
    with tf.name_scope('de_uint2_bn1') as scope:
        b_norm=bn(de_uint2_conv1, training,0,noise_std,15)
        de_uint2_bn1= tf.nn.relu(b_norm,name=scope)   
    with tf.name_scope('de_uint2_conv0') as scope:  
        kernel = weight_variable([1, 1, 64, 64])
        output_shape=tf.shape(cor_conv4)
        deconv = de_conv2d(de_uint2_bn1, kernel,output_shape)
        biases = bias_variable([64])
        de_uint2_conv0 = tf.nn.bias_add(deconv, biases)       
    with tf.name_scope('de_uint2_bn0') as scope:
        de_uint2_bn0=bn(de_uint2_conv0, training,0,noise_std,16)     
    with tf.name_scope('de_residual2_sum') as scope:
        de_residual2_sum=de_uint2_bn0+defc1
        de_residual2_sum= tf.nn.relu(de_residual2_sum,name=scope)
        
    #de_adapt   
    with tf.name_scope('deconv4') as scope:  
        kernel = weight_variable([1, 1, 32, 64])
        output_shape=tf.shape(cor_conv3)
        deconv = de_conv2d(de_residual2_sum, kernel,output_shape)
        biases = bias_variable([32])
        deconv4 = tf.nn.bias_add(deconv, biases)
    with tf.name_scope('debn4') as scope:
        debn4=bn(deconv4, training,0,noise_std,17)
#        debn4_r= tf.nn.relu(debn4,name=scope)
    with tf.name_scope('de_uint1_conv2') as scope:  
        kernel = weight_variable([1, 1, 32, 32])
        output_shape=tf.shape(cor_conv3)
        deconv = de_conv2d(debn4, kernel,output_shape)
        biases = bias_variable([32])
        de_uint1_conv2 = tf.nn.bias_add(deconv, biases)
    with tf.name_scope('de_uint1_bn2') as scope:
        b_norm=bn(de_uint1_conv2, training,0,noise_std,18)
        de_uint1_bn2= tf.nn.relu(b_norm,name=scope)   
    with tf.name_scope('de_uint1_conv1') as scope:  
        kernel = weight_variable([1, 3, 32, 32])
        output_shape=tf.shape(cor_conv3)
        deconv = de_conv2d_p(de_uint1_bn2, kernel,output_shape)
        biases = bias_variable([32])
        de_uint1_conv1 = tf.nn.bias_add(deconv, biases)
    with tf.name_scope('de_uint1_bn1') as scope:
        b_norm=bn(de_uint1_conv1, training,0,noise_std,19)
        de_uint1_bn1= tf.nn.relu(b_norm,name=scope)   
    with tf.name_scope('de_uint1_conv0') as scope:  
        kernel = weight_variable([1, 1, 32, 32])
        output_shape=tf.shape(cor_conv3)
        deconv = de_conv2d(de_uint1_bn1, kernel,output_shape)
        biases = bias_variable([32])
        de_uint1_conv0 = tf.nn.bias_add(deconv, biases) 
    with tf.name_scope('de_uint1_bn0') as scope:
        de_uint1_bn0=bn(de_uint1_conv0, training,0,noise_std,20)
    with tf.name_scope('de_residual1_sum') as scope:
        de_residual1_sum=de_uint1_bn0+debn4
        de_residual1_sum= tf.nn.relu(de_residual1_sum,name=scope)
    
    #de_adapt   
    with tf.name_scope('deconv3') as scope:  
        kernel = weight_variable([1, 1, 16, 32])
        output_shape=tf.shape(cor_conv2)
        deconv = de_conv2d_s2(de_residual1_sum, kernel,output_shape)
        biases = bias_variable([16])
        deconv3 = tf.nn.bias_add(deconv, biases)
    with tf.name_scope('debn3') as scope:
        debn3=bn(deconv3, training,0,noise_std,21)
#        debn3_r= tf.nn.relu(debn3,name=scope)
    with tf.name_scope('de_uint0_conv2') as scope:  
        kernel = weight_variable([1, 1, 16, 16])
        output_shape=tf.shape(cor_conv2)
        deconv = de_conv2d(debn3, kernel,output_shape)
        biases = bias_variable([16])
        de_uint0_conv2 = tf.nn.bias_add(deconv, biases)
    with tf.name_scope('de_uint0_bn2') as scope:
        b_norm=bn(de_uint0_conv2, training,0,noise_std,22)
        de_uint0_bn2= tf.nn.relu(b_norm,name=scope)   
    with tf.name_scope('de_uint0_conv1') as scope:  
        kernel = weight_variable([1, 3, 16, 16])
        output_shape=tf.shape(cor_conv2)
        deconv = de_conv2d_p(de_uint0_bn2, kernel,output_shape)
        biases = bias_variable([16])
        de_uint0_conv1 = tf.nn.bias_add(deconv, biases)
    with tf.name_scope('de_uint0_bn1') as scope:
        b_norm=bn(de_uint0_conv1, training,0,noise_std,23)
        de_uint0_bn1= tf.nn.relu(b_norm,name=scope)   
    with tf.name_scope('de_uint0_conv0') as scope:  
        kernel = weight_variable([1, 1, 16, 16])
        output_shape=tf.shape(cor_conv2)
        deconv = de_conv2d(de_uint0_bn1, kernel,output_shape)
        biases = bias_variable([16])
        de_uint0_conv0 = tf.nn.bias_add(deconv, biases)  
    with tf.name_scope('de_uint0_bn0') as scope:
        de_uint0_bn0=bn(de_uint0_conv0, training,0,noise_std,24)
    with tf.name_scope('de_residual0_sum') as scope:
        de_residual0_sum=de_uint0_bn0+debn3
        de_residual0_sum= tf.nn.relu(de_residual0_sum,name=scope)  
    #de_adapt   

    with tf.name_scope('deconv2') as scope:  
        kernel = weight_variable([1, 1, 16, 16])
        output_shape=tf.shape(cor_conv1)
        deconv = de_conv2d(de_residual0_sum, kernel,output_shape)
        biases = bias_variable([16])
        deconv2 = tf.nn.bias_add(deconv, biases)
    with tf.name_scope('debn2') as scope:
        b_norm=bn(deconv2, training,0,noise_std,25)
        debn2= tf.nn.relu(b_norm,name=scope)
    with tf.name_scope('deconv1') as scope:
        output=tf.shape(re_data)
        kernel = weight_variable([1, 7, 1, 16]) 
        deconv = de_conv2d_s2(debn2, kernel,output)
        biases = bias_variable([1])
        bias = tf.nn.bias_add(deconv, biases)
        deconv1=tf.nn.relu(bias, name=scope)
        reconstruction=deconv1
    d_conv['labeled']['z'][0], d_conv['unlabeled']['z'][0] = split_lu(reconstruction)
    d_conv['labeled']['z'][1], d_conv['unlabeled']['z'][1] = split_lu(debn2)
    d_conv['labeled']['z'][2], d_conv['unlabeled']['z'][2] = split_lu(de_residual0_sum)
    d_conv['labeled']['z'][3], d_conv['unlabeled']['z'][3] = split_lu(de_residual1_sum)
    d_conv['labeled']['z'][4], d_conv['unlabeled']['z'][4] = split_lu(de_residual2_sum)
    return d_conv




print ("=== Clean Encoder ===")
y, clean,clean_inputs,conv1,conv2,conv3,conv4,fc1,fc2 = encoder(re_inputs, 0.0)  # 0.0 -> do not add noise

print ("=== Corrupted Encoder ===")
y_c, corr,corrupted_inputs,cor_conv1,cor_conv2,cor_conv3,cor_conv4,cor_fc1,cor_fc2 = encoder(re_inputs, noise_std)


print ("=== Decoder ===")
d_conv=decoder(corrupted_inputs,cor_conv1,cor_conv2,cor_conv3,cor_conv4,cor_fc1,cor_fc2)

u_cost=[]
for l in [0,1,2,3,4]:
    if l==0:
        z=clean['unlabeled']['z'][l]
        data_shape=tf.shape(z)
        z=tf.reshape(z,[-1,data_shape[1]*data_shape[2]*data_shape[3]])
        z_c=d_conv['unlabeled']['z'][l]
        data_shape=tf.shape(z_c)
        z_c=tf.reshape(z_c,[-1,data_shape[1]*data_shape[2]*data_shape[3]])  
    elif l==5:
        z=clean['unlabeled']['z'][l]
        z_c=tf.transpose(d_conv['unlabeled']['z'][l])   
    else:
        z=clean['unlabeled']['z'][l]
        data_shape=tf.shape(z)
        z=tf.reshape(z,[data_shape[0]*data_shape[1],-1])
        
        z_c=d_conv['unlabeled']['z'][l]
        data_shape=tf.shape(z_c)
        z_c=tf.reshape(z_c,[data_shape[0]*data_shape[1],-1])

    u_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_c - z), 1)) / layer_sizes[l]) * denoising_cost[l])


u_cost = tf.add_n(u_cost)
y_N = labeled_fc(y_c)

cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y_N), 1))  # supervised cost


loss = 0.8*cost + 0.2*u_cost  # total cost





correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(outputs, 1))  # no of correct predictions

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)

learning_rate = tf.Variable(starter_learning_rate, trainable=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


bn_updates = tf.group(*bn_assigns)
with tf.control_dependencies([train_step]):
    train_step = tf.group(bn_updates)

print ("===  Loading Data ===")
mnist = input_data_paviau.read_data_sets(n_labeled=num_labeled, one_hot=True)

saver = tf.train.Saver()

print ("===  Starting Session ===")
sess = tf.Session()

i_iter = 0

ckpt = tf.train.get_checkpoint_state('checkpoints/') 
if ckpt and ckpt.model_checkpoint_path:
    # if checkpoint exists, restore the parameters and set epoch_n and i_iter
    saver.restore(sess, ckpt.model_checkpoint_path)
    epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
    i_iter = int((epoch_n+1) * (num_examples/batch_size))
    print ("Restored Epoch ", epoch_n)
else:
    # no checkpoint exists. create checkpoints directory if it does not exist.
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    init = tf.global_variables_initializer()
    sess.run(init)

print ("=== Training ===")

print ("Initial Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False}), "%")
   
for i in tqdm(range(i_iter, num_iter)):
    images, labels = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={inputs: images, outputs: labels, training: True})
    if (i > 1) and (int((i+1) % (num_iter/num_epochs) )== 0):
        epoch_n = int(i/(num_examples/batch_size))
        if (epoch_n+1) >= decay_after:
            ratio = 1.0 * (num_epochs - (epoch_n+1))  # epoch_n + 1 because learning rate is set for next epoch
            ratio = max(0, ratio / (num_epochs - decay_after))
            sess.run(learning_rate.assign(starter_learning_rate * ratio))
        saver.save(sess, 'checkpoints/model.ckpt', epoch_n)
        with open('train_log', 'a') as train_log:
            # write test accuracy to file "train_log"
            train_log_w = csv.writer(train_log)
            log_i = [epoch_n] + sess.run([accuracy], feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False})
            train_log_w.writerow(log_i)

print ("Final Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False}), "%")

sess.close()