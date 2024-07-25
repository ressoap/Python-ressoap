import numpy as np

# 加载原始数据
inputs = np.loadtxt('C:/Users/ressoap/PycharmProjects/py深度学习作业/test/c_2_input.txt')
Ture = np.loadtxt('C:/Users/ressoap/PycharmProjects/py深度学习作业/test/train.txt')

# 切片和重塑数据

print("调整后的输入数据形状:", inputs.shape)
Ture = Ture[:30]
# 保存调整后的数据
np.savetxt('c_2_input_adjusted.txt', inputs)
np.savetxt('train_adjusted.txt', Ture)

