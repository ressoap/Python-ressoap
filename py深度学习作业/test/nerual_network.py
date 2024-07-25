import numpy as np
def tanh(x):
	return np.tanh(x)
def tanh2deriv(output):
	return 1 - (output ** 2)




inputs = np.loadtxt('c_2_input_adjusted.txt')
# inputs = inputs.reshape(30, 20)
# shape=inputs.shape
# print(shape)
Ture = np.loadtxt('train_adjusted.txt')
Ture = Ture.reshape(30, 1)
# shape=Ture.shape
# print(shape)
np.random.seed(1)
# relu = lambda x: (x >= 0) * x
# relu2deriv = lambda x: x >= 0
alpha = 0.00006
iterations = 100
w_0 = 20
w_1 = 500
w_2 = 1000
w_3 = 750
w_4 = 500
w_5 = 250
w_6 = 500
w_7 = 250
w_8 = 500
w_9 = 750
w_10 = 500
w_11 = 250
w_12 = 1
# 批量
batch_size = 30
weights_0_1 = 0.4 * np.random.random((w_0, w_1)) - 0.2
weights_1_2 = 0.4 * np.random.random((w_1, w_2)) - 0.2
weights_2_3 = 0.4 * np.random.random((w_2, w_3)) - 0.2
weights_3_4 = 0.4 * np.random.random((w_3, w_4)) - 0.2
weights_4_5 = 0.4 * np.random.random((w_4, w_5)) - 0.2
weights_5_6 = 0.4 * np.random.random((w_5, w_6)) - 0.2
weights_6_7 = 0.4 * np.random.random((w_6, w_7)) - 0.2
weights_7_8 = 0.4 * np.random.random((w_7, w_8)) - 0.2
weights_8_9 = 0.4 * np.random.random((w_8, w_9)) - 0.2
weights_9_10 = 0.4 * np.random.random((w_9, w_10)) - 0.2
weights_10_11 = 0.4 * np.random.random((w_10, w_11)) - 0.2
weights_11_12 = 0.4 * np.random.random((w_11, w_12)) - 0.2
for j in range(iterations):
	layer6_error = 0
	for i in range(int(len(inputs) / batch_size)):
		# ！！！！！！
		batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))

		layer_0 = inputs[batch_start:batch_end]

		layer_1 = tanh(np.dot(layer_0, weights_0_1))
		dropout_mask_1 = np.random.randint(2, size=layer_1.shape)
		layer_1 *= dropout_mask_1 * 2

		layer_2 = tanh(np.dot(layer_1, weights_1_2))
		dropout_mask_2 = np.random.randint(2, size=layer_2.shape)
		layer_2 *= dropout_mask_2 * 2

		layer_3 = tanh(np.dot(layer_2, weights_2_3))
		dropout_mask_3 = np.random.randint(2, size=layer_3.shape)
		layer_3 *= dropout_mask_3 * 2

		layer_4 = tanh(np.dot(layer_3, weights_3_4))
		dropout_mask_4 = np.random.randint(2, size=layer_4.shape)
		layer_4 *= dropout_mask_4 * 2

		layer_5 = tanh(np.dot(layer_4, weights_4_5))
		dropout_mask_5 = np.random.randint(2, size=layer_5.shape)
		layer_5 *= dropout_mask_5 * 2

		layer_6 = tanh(np.dot(layer_5, weights_5_6))

		layer_7 = tanh(np.dot(layer_6, weights_6_7))

		layer_8 = tanh(np.dot(layer_7, weights_7_8))

		layer_9 = tanh(np.dot(layer_8, weights_8_9))

		layer_10 = tanh(np.dot(layer_9, weights_9_10))
		layer_11 = tanh(np.dot(layer_10, weights_10_11))
		layer_12 = np.dot(layer_11, weights_11_12)
		# shape3=layer_12.shape
		# print(shape3)
		# shape4=Ture[batch_start:batch_end].shape
		# print(shape4)
		error = np.sum(layer_12 - Ture[batch_start:batch_end])
		# print(layer6_error)
		layer_12_delta = (layer_12 - Ture[batch_start:batch_end]) / batch_size
		layer_11_delta = np.dot(layer_12_delta, weights_11_12.T) * tanh2deriv(layer_11)
		layer_10_delta = np.dot(layer_11_delta, weights_10_11.T) * tanh2deriv(layer_10)
		layer_9_delta = np.dot(layer_10_delta, weights_9_10.T) * tanh2deriv(layer_9)
		layer_8_delta = np.dot(layer_9_delta, weights_8_9.T) * tanh2deriv(layer_8)
		layer_7_delta = np.dot(layer_8_delta, weights_7_8.T) * tanh2deriv(layer_7)
		layer_6_delta = np.dot(layer_7_delta, weights_6_7.T) * tanh2deriv(layer_6)
		layer_5_delta = np.dot(layer_6_delta, weights_5_6.T) * tanh2deriv(layer_5)
		layer_5_delta *= dropout_mask_5
		layer_4_delta = np.dot(layer_5_delta, weights_4_5.T) * tanh2deriv(layer_4)
		layer_4_delta *= dropout_mask_4
		layer_3_delta = np.dot(layer_4_delta, weights_3_4.T) * tanh2deriv(layer_3)
		layer_3_delta *= dropout_mask_3
		layer_2_delta = np.dot(layer_3_delta, weights_2_3.T) * tanh2deriv(layer_2)
		layer_2_delta *= dropout_mask_2
		layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * tanh2deriv(layer_1)
		layer_1_delta *= dropout_mask_1
		weights_11_12 -= alpha * layer_11.T.dot(layer_12_delta)
		weights_10_11 -= alpha * layer_10.T.dot(layer_11_delta)
		weights_9_10 -= alpha * layer_9.T.dot(layer_10_delta)
		weights_8_9 -= alpha * layer_8.T.dot(layer_9_delta)
		weights_7_8 -= alpha * layer_7.T.dot(layer_8_delta)
		weights_6_7 -= alpha * layer_6.T.dot(layer_7_delta)
		weights_5_6 -= alpha * layer_5.T.dot(layer_6_delta)
		weights_4_5 -= alpha * layer_4.T.dot(layer_5_delta)
		weights_3_4 -= alpha * layer_3.T.dot(layer_4_delta)
		weights_2_3 -= alpha * layer_2.T.dot(layer_3_delta)
		weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
		weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)
		print("Epochs: " + str(j) + "  Train_Error: " + str(error))
total=[]
total.append(weights_0_1)
total.append(weights_1_2)
total.append(weights_2_3)
total.append(weights_3_4)
total.append(weights_4_5)
total.append(weights_5_6)
total.append(weights_6_7)
total.append(weights_7_8)
total.append(weights_8_9)
total.append(weights_9_10)
total.append(weights_10_11)
total.append(weights_11_12)
np.save('weights.npy',total)


