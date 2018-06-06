# -*- coding: utf-8 -*-


import numpy as np
import scipy.special


class NeuralNetwork(object):
	def __init__(self, inputnodes, hiddennodes, outputnodes,lr):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
		self.lr = lr
		self.activation_func = lambda x: scipy.special.expit(x)

	def train(self, input_list, target_list):
		inputs = np.array(input_list, ndmin=2).T
		targets = np.array(target_list, ndmin=2).T
		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_func(hidden_inputs)
		finnal_inputs = np.dot(self.who, hidden_outputs)
		finnal_outputs = self.activation_func(finnal_inputs)

		output_errors = targets - finnal_outputs
		hidden_errors = np.dot(self.who.T, output_errors)

		self.who += self.lr * np.dot((output_errors * finnal_outputs * (1.0 - finnal_outputs)), np.transpose(hidden_outputs))
		self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

	def query(self, input_list):
		inputs = np.array(input_list, ndmin=2).T
		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_func(hidden_inputs)
		finnal_inputs = np.dot(self.who, hidden_outputs)
		finnal_outputs = self.activation_func(finnal_inputs)
		return finnal_outputs


