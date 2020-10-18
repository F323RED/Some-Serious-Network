import numpy as np
import F323REDsMLKit as ml
from NMISTLoader import Loader
from TwoLayerNet import TwoLayerNet

# Path setting
path = r"D:/程式碼/Python/Deep Learning Projects/MNIST dataset/"
load = Loader(path + "t10k-images-idx3-ubyte.gz", \
    path + "t10k-labels-idx1-ubyte.gz", \
    path + "train-images-idx3-ubyte.gz", \
    path + "train-labels-idx1-ubyte.gz")

# Load MNIST dataset
print("Loading...")
(t_i, t_l), (x_i, x_l) = load.LoadData()

x_image = np.array(x_i)
x_label = np.array(x_l)
t_image = np.array(t_i)
t_label = np.array(t_l)
print("Done")

# Set parameters
print("Set parameters.")
net = TwoLayerNet(input_size = 784, hidden_size = 100, output_size=10)
iter_count = 10000
train_size = len(x_label)
batch_size = 100
learning_rate = 0.1

# Calculate accuracy
a = np.array(x_image[0 : 100])
print(a.shape)

b = np.array(x_label[0 : 100])
print(b.shape)
c = np.zeros((b.size, 10))
c[np.arange(b.size), b] = 1         # One-hot array

print("Accuracy before:")
print(net.Accuracy(a, c))

# Start
print("Start training.")
for i in range(iter_count):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_image[batch_mask]
    a = x_label[batch_mask]
    y_batch = np.zeros((a.size, 10))
    y_batch[np.arange(a.size), a] = 1       # One-hot array

    grad = net.Gradient(x_batch, y_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        net.parmas[key] -= learning_rate * grad[key]
        
    print("{:>6.4f} loss".format(net.Loss(x_batch, y_batch)))
    print("{:>6d}/{:>6d} Done".format(i, iter_count))

## Calculate Accuracy
a = np.array(t_image[0 : 100])

b = np.array(t_label[0 : 100])
c = np.zeros((b.size, 10))
c[np.arange(b.size), b] = 1         # One-hot array

print("Accuracy after:")
print(net.Accuracy(a, c))

