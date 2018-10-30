import numpy as np

class Softmax_Layer():
    def __init__(self):
        pass
    def forward(self,input):
        m = input.shape[0]
        input = input - (np.max(input,axis=1)).reshape(m,1)
        e_op = np.exp(input)
        sum_op = (e_op.sum(axis=1)).reshape(m,1)
        output = e_op / sum_op
        return output
    def backward(self,input,targets):
        output = self.forward(input)
        grad_output = output - targets
        return grad_output

class Dense_Layer():
    def __init__(self,num_input,num_output,learning_rate=0.01,weight=-0.5,bias=-0.5):
        self.learning_rate = learning_rate
        self.weights = np.random.randn(num_input,num_output)*(weight)
        self.biases = np.random.randn(1,num_output)*(bias)
    def get_biases(self):
        return self.biases
    def get_weights(self):
        return self.weights
    def forward(self,input):
        return np.dot(input,self.weights) + self.biases
    def backward(self,input,grad_output):
        grad_input = np.dot(grad_output,self.weights.T)
        grad_weights = np.dot(input.T,grad_output)/input.shape[0]
        grad_biases = grad_output.mean(axis=0)
        self.weights = self.weights - self.learning_rate*grad_weights
        self.biases = self.biases - self.learning_rate*grad_biases
        return grad_input

class Tanh_Layer():
    def __init__(self):
        pass
    def _tanh(self,input):
        return np.tanh(input)
    def forward(self,input):
        return self._tanh(input)
    def backward(self,input,grad_output):
        tanh_grad = 1 - (self._tanh(input))**2
        return grad_output*tanh_grad

class ReLU_Layer():
    def __init__(self):
        pass
    def forward(self,input):
        return np.maximum(0,input)
    def backward(self,input,grad_output):
        relu_grad = input > 0
        return grad_output * relu_grad

class Sigmoid_Layer():
    def __init(self):
        pass
    def _sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
    def forward(self,input):
        return self._sigmoid(input)
    def backward(self,input,grad_output):
        sigmoid_grad = self._sigmoid(input)*(1-self._sigmoid(input))
        return grad_output*sigmoid_grad

def get_net_config():
    net_config = []
    num_layer = 0
    while num_layer < 2:
        num_layer = int(input("需要几层？\n"))
        if num_layer < 2:
            print("至少两层！")
    num_layer = int(num_layer) - 1
    num_input = input("输入层几个神经元？\n")
    net_config.append(int(num_input))
    for i in range(num_layer-1):
        hint_mes = "隐藏层第" + str(i+1) + "层需要几个神经元？\n"
        num_hidden = input(hint_mes)
        net_config.append(int(num_hidden))
        hint_mes = "这一个激活层用什么激活函数？（sigmoid 0,tanh 1,relu 2, softmax 3）\n"
        activation_type = input(hint_mes)
        net_config.append(int(activation_type))
    num_output = input("输出层需要几个神经元？\n")
    net_config.append(int(num_output))
    has_softmax = input("是否要加上一层sofmax层？ -2 否，-1是\n")
    net_config.append(int(has_softmax))
    #print(net_config)
    return net_config

class Network:
    def __init__(self,net_config,learning_rate=0.01,weight=-0.5,bias=-0.5):
        self.net_config = net_config
        self.network = []
        self.learning_rate = learning_rate
        prev = net_config[0]
        current = net_config[1]
        i = 1
        while i < len(net_config):
            self.network.append(Dense_Layer(prev,current,learning_rate,weight,bias))
            print("Append dense layer")
            if i+1 < len(net_config):
                activation_type = net_config[i+1]
                if activation_type == 0:
                    self.network.append(Sigmoid_Layer())
                    print("append sigmoid layer")
                elif activation_type == 1:
                    self.network.append(Tanh_Layer())
                    print("append tanh layer")
                elif activation_type == 2:
                    self.network.append(ReLU_Layer())
                    print("append relu layer")
                elif activation_type == -1:
                    self.network.append(Softmax_Layer())
                    print("append softmax layer")
                else :
                    raise Exception("invalid net config")
                    pass
            i = i + 2
            prev = current
            if i < len(net_config):
                current = net_config[i]
    def forward(self,input):
        activations = []
        for layer in self.network:
            activation = layer.forward(input)
            activations.append(activation)
            input = activations[-1]

        assert len(activations) == len(self.network)
        return activations

    def train(self,X,y):
        layer_activations = self.forward(X)
        print("target dimension: " , y.shape[1], "output dimension: ",  layer_activations[-1].shape[-1])
        if y.shape[1] != layer_activations[-1].shape[-1]:
            raise Exception("dimension not equal!!!")
        layer_inputs = [X] + layer_activations
        logits = layer_activations[-1]

        if self.net_config[-1] == -1:
            [loss,loss_grad] = self._cross_entropy_function(logits,y)
        else:
            [loss,loss_grad] = self._square_error_function(logits,y)

        for layer_i in range(len(self.network))[::-1]:
            layer = self.network[layer_i]
            loss_grad = layer.backward(layer_inputs[layer_i],loss_grad)
        return loss/X.shape[0]
    def predict(self,X):
        logits = self.forward(X)[-1]
        return logits
    def _square_error_function(self,output,targets):
        loss = np.square(output - targets).sum()
        loss_grad = 2.0*(output - targets)
        return [loss,loss_grad]
        
    def _cross_entropy_function(self,output,targets):
        loss = -((targets*np.log(output)).sum())
        loss_grad = targets
        return [loss,loss_grad]
        
    def get_weights(self):
        layer_i = 0
        weights = []
        while layer_i < len(self.network):
            weight = self.network[layer_i].get_weights()
            weights.append(weight)
            layer_i = layer_i + 2
        return weights
    def get_biases(self):
        layer_i = 0
        biases = []
        while layer_i < len(self.network):
            bias = self.network[layer_i].get_biases()
            biases.append(bias)
            layer_i = layer_i + 2
        return biases

