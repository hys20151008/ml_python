# -*- coding: utf-8 -*-


from nw import NeuralNetwork
import numpy as np

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

lr = 0.1

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes,lr)

trainning_data = open("mnist_dataset/mnist_train_100.csv", 'r')
trainning_data_list = trainning_data.readlines()
trainning_data.close()

#train
with open("mnist_dataset/mnist_train.csv", 'r') as f:
	for elt in f:
	    v = elt.split(',')
	    inputs = (np.asfarray(v[1:])/255.0 * 0.99) + 0.01
	    targets = np.zeros(output_nodes) + 0.01
	    targets[int(v[0])] = 0.99
	    nn.train(inputs, targets)


# test
score = []
count_all = 0
count_correct = 0
with open("mnist_dataset/mnist_test.csv", 'r') as f:
	for elt in f:
		count_all += 1
		v = elt.split(',')
		correct_label = int(v[0])
		print("correct label is %d" % correct_label)
		inputs = (np.asfarray(v[1:])/255.0 *0.99)+0.01
		outputs = nn.query(inputs)
		label = np.argmax(outputs)
		print("network answer is %d" % label)
		if(label == correct_label):
			count_correct +=1
		
print(float(count_correct/count_all))
