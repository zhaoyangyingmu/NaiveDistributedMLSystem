import numpy as np
import matplotlib.pyplot as plt
from network_elements import Network

## set up network
print("假如跑崩了，再跑一次就好了。")
net_config = [1,30,2,1]
learning_rate = 0.1
network_para = Network(net_config,learning_rate)
network_worker = Network(net_config,learning_rate)

## training
x_train = np.linspace(-np.pi, np.pi, 600).reshape(600,-1)
y_train = np.sin(x_train)

ll = []
for e in range(3000):
    weights = network_para.get_weights()
    biases = network_para.get_biases()
    layer_activations = network_worker.forward(weights, biases, x_train)
    loss = network_para.train(x_train,y_train, layer_activations)
    ll.append(loss)
x = np.arange(1,len(ll)+1)
plt.title('loss with three layer, hidden layer = 30')
label_str = ' activation = tanh'
plt.plot(x[100:], ll[100:],label=(label_str))

plt.legend()
plt.xlabel('iteration times')
plt.ylabel('loss')
plt.show()
## testing
x_test = np.linspace(-np.pi, np.pi, 100).reshape(100,-1)
y_test = np.sin(x_test)


y_predit = network.predict(x_test)
loss = np.square(y_predit - y_test).sum() / x_test.shape[0]
print("test loss is " , loss, "with layer size = ",30)
