
# Lab 1: Intro to PyTorch and Music Generation with RNNs

# In this lab, you'll get exposure to using PyTorch and learn how it can be used for deep learning.
# Go through the code and run each cell.
# Along the way, you'll encounter several TODO blocks -- follow the instructions to fill them out before running those cells and continuing.
#
# Part 1: Intro to PyTorch 0.1 Install PyTorch PyTorch is a popular deep learning library known for its flexibility
# and ease of use. Here we'll learn how computations are represented and how to define a simple neural network in
# PyTorch. For all the labs in Introduction to Deep Learning 2025, there will be a PyTorch version available.

import torch
import torch.nn as nn

import mitdeeplearning as mdl


import numpy as np
import matplotlib.pyplot as plt



# 1.1 What is PyTorch?
# PyTorch is a machine learning library, like TensorFlow. At its core, PyTorch provides an interface for creating and manipulating tensors, which are data structures that you can think of as multi-dimensional arrays. Tensors are represented as n-dimensional arrays of base datatypes such as a string or integer -- they provide a way to generalize vectors and matrices to higher dimensions. PyTorch provides the ability to perform computation on these tensors, define neural networks, and train them efficiently.
#
# The shape of a PyTorch tensor defines its number of dimensions and the size of each dimension. The ndim or dim of a PyTorch tensor provides the number of dimensions (n-dimensions) -- this is equivalent to the tensor's rank (as is used in TensorFlow), and you can also think of this as the tensor's order or degree.
#
# Let’s start by creating some tensors and inspecting their properties:

integer = torch.tensor(1234)
decimal = torch.tensor(3.14159265359)

print(f"`integer` is a {integer.ndim}-d Tensor: {integer}")
print(f"`decimal` is a {decimal.ndim}-d Tensor: {decimal}")

# Vectors and lists can be used to create 1-d tensors:

fibonacci = torch.tensor([1, 1, 2, 3, 5, 8])
count_to_100 = torch.tensor(range(100))

print(f"`fibonacci` is a {fibonacci.ndim}-d Tensor with shape: {fibonacci.shape}")
print(f"`count_to_100` is a {count_to_100.ndim}-d Tensor with shape: {count_to_100.shape}")

# Next, let’s create 2-d (i.e., matrices) and higher-rank tensors. In image processing and computer vision, we will use 4-d Tensors with dimensions corresponding to batch size, number of color channels, image height, and image width.

### Defining higher-order Tensors ###

'''TODO: Define a 2-d Tensor'''
matrix = torch.tensor(
    [[1, 1, 2, 3, 5, 8],
     [1, 1, 2, 3, 5, 8],
     [1, 1, 2, 3, 5, 8],
     [1, 1, 2, 3, 5, 8]])

assert isinstance(matrix, torch.Tensor), "matrix must be a torch Tensor object"
assert matrix.ndim == 2

'''TODO: Define a 4-d Tensor.'''
# Use torch.zeros to initialize a 4-d Tensor of zeros with size 10 x 3 x 256 x 256.
#   You can think of this as 10 images where each image is RGB 256 x 256.
images = torch.zeros(10,3,256,256)

print(f"images is a {images.ndim}-d Tensor with shape: {images.shape}")

assert isinstance(images, torch.Tensor), "images must be a torch Tensor object"
assert images.ndim == 4, "images must have 4 dimensions"
assert images.shape == (10, 3, 256, 256), "images is incorrect shape"


# As you have seen, the shape of a tensor provides the number of elements in each tensor dimension. The shape is quite useful, and we'll use it often. You can also use slicing to access subtensors within a higher-rank tensor:

row_vector = matrix[1]
column_vector = matrix[:, 1]
scalar = matrix[0, 1]

print(f"`row_vector`: {row_vector}")
print(f"`column_vector`: {column_vector}")
print(f"`scalar`: {scalar}")

# 1.2 Computations on Tensors
# A convenient way to think about and visualize computations in a machine learning framework like PyTorch is in terms of graphs. We can define this graph in terms of tensors, which hold data, and the mathematical operations that act on these tensors in some order. Let's look at a simple example, and define this computation using PyTorch:

# Create the nodes in the graph and initialize values
a = torch.tensor(15)
b = torch.tensor(61)

# Add them!
c1 = torch.add(a, b)
c2 = a + b  # PyTorch overrides the "+" operation so that it is able to act on Tensors
print(f"c1: {c1}")
print(f"c2: {c2}")

# Notice how we've created a computation graph consisting of PyTorch operations, and how the output is a tensor with value 76 -- we've just created a computation graph consisting of operations, and it's executed them and given us back the result.
#
# Now let's consider a slightly more complicated example:


# Here, we take two inputs, a, b, and compute an output e. Each node in the graph represents an operation that takes some input, does some computation, and passes its output to another node.
#
# Let's define a simple function in PyTorch to construct this computation function:

### Defining Tensor computations ###

# Construct a simple computation function
def func(a, b):
    '''TODO: Define the operation for c, d, e.'''
    c = torch.add(a, b)
    d = torch.subtract(b, 1)
    e = torch.mul(c,d)
    return e

# Now, we can call this function to execute the computation graph given some inputs a,b:

# Consider example values for a,b
a, b = 1.5, 2.5
# Execute the computation
e_out = func(a, b)
print(f"e_out: {e_out}")

# 1.3 Neural networks in PyTorch
# We can also define neural networks in PyTorch. PyTorch uses torch.nn.Module, which serves as a base class for all neural network modules in PyTorch and thus provides a framework for building and training neural networks.
#
# Let's consider the example of a simple perceptron defined by just one dense (aka fully-connected or linear) layer:
# ...


# We will use torch.nn.Module to define layers -- the building blocks of neural networks. Layers implement common neural networks operations. In PyTorch, when we implement a layer, we subclass nn.Module and define the parameters of the layer as attributes of our new class. We also define and override a function forward, which will define the forward pass computation that is performed at every step. All classes subclassing nn.Module should override the forward function.
#
# Let's write a dense layer class to implement a perceptron defined above.


### Defining a dense layer ###

# num_inputs: number of input nodes
# num_outputs: number of output nodes
# x: input to the layer

class OurDenseLayer(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(OurDenseLayer, self).__init__()
        # Define and initialize parameters: a weight matrix W and bias b
        # Note that the parameter initialize is random!
        self.W = torch.nn.Parameter(torch.randn(num_inputs, num_outputs))
        self.bias = torch.nn.Parameter(torch.randn(num_outputs))

    def forward(self, x):
        '''TODO: define the operation for z (hint: use torch.matmul).'''
        z = torch.matmul(x,self.W) + self.bias

        '''TODO: define the operation for out (hint: use torch.sigmoid).'''
        y = torch.sigmoid(z)
        return y

# Now, let's test the output of our layer.

# Define a layer and test the output!
num_inputs = 2
num_outputs = 3
layer = OurDenseLayer(num_inputs, num_outputs)
x_input = torch.tensor([[1, 2.]])
y = layer(x_input)

print(f"input shape: {x_input.shape}")
print(f"output shape: {y.shape}")
print(f"output result: {y}")


# Conveniently, PyTorch has defined a number of nn.Modules (or Layers) that are commonly used in neural networks, for example a nn.Linear or nn.Sigmoid module.
#
# Now, instead of using a single Module to define our simple neural network, we'll use the nn.Sequential module from PyTorch and a single nn.Linear layer to define our network. With the Sequential API, you can readily create neural networks by stacking together layers like building blocks.


### Defining a neural network using the PyTorch Sequential API ###

### Defining a neural network using the PyTorch Sequential API ###

# define the number of inputs and outputs
n_input_nodes = 2
n_output_nodes = 3

# Define the model
'''TODO: Use the Sequential API to define a neural network with a
    single linear (dense!) layer, followed by non-linearity to compute z'''
model = nn.Sequential(
    # linear layer with input size 2 and output size 3
    nn.Linear(n_input_nodes, n_output_nodes),
    # Sigmoid activation function
    nn.Sigmoid()
)

# We've defined our model using the Sequential API. Now, we can test it out using an example input:

# Test the model with example input
x_input = torch.tensor([[1, 2.]])
model_output = model(x_input)
print(f"input shape: {x_input.shape}")
print(f"output shape: {y.shape}")
print(f"output result: {y}")

# With PyTorch, we can create more flexible models by subclassing nn.Module. The nn.Module class allows us to group layers together flexibly to define new architectures.
#
# As we saw earlier with OurDenseLayer, we can subclass nn.Module to create a class for our model, and then define the forward pass through the network using the forward function. Subclassing affords the flexibility to define custom layers, custom training loops, custom activation functions, and custom models. Let's define the same neural network model as above (i.e., Linear layer with an activation function after it), now using subclassing and using PyTorch's built in linear layer from nn.Linear.


### Defining a model using subclassing ###

class LinearWithSigmoidActivation(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearWithSigmoidActivation, self).__init__()
        '''TODO: define a model with a single Linear layer and sigmoid activation.'''
        self.linear = nn.Linear(num_inputs, num_outputs)
        self.activation = nn.Sigmoid()

    def forward(self, inputs):
        linear_output = self.linear(inputs)
        output = self.activation(linear_output)
        return output

# Let's test out our new model, using an example input, setting n_input_nodes=2 and n_output_nodes=3 as before.

n_input_nodes = 2
n_output_nodes = 3
model = LinearWithSigmoidActivation(n_input_nodes, n_output_nodes)
x_input = torch.tensor([[1, 2.]])
y = model(x_input)
print(f"input shape: {x_input.shape}")
print(f"output shape: {y.shape}")
print(f"output result: {y}")


# Importantly, nn.Module affords us a lot of flexibility to define custom models. For example, we can use boolean arguments in the forward function to specify different network behaviors, for example different behaviors during training and inference. Let's suppose under some instances we want our network to simply output the input, without any perturbation. We define a boolean argument isidentity to control this behavior:


### Custom behavior with subclassing nn.Module ###

class LinearButSometimesIdentity(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearButSometimesIdentity, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, inputs, isidentity=False):
        """Implement the behavior where the network outputs the input, unchanged,
            under control of the isidentity argument."""
        if isidentity:
            return  inputs
        else:
            return self.linear(inputs)

# Test the IdentityModel
model = LinearButSometimesIdentity(num_inputs=2, num_outputs=3)
x_input = torch.tensor([[1, 2.]])

# Let's test this behavior:

'''TODO: pass the input into the model and call with and without the input identity option.'''
out_with_linear =model(x_input)

out_with_identity = model(x_input,isidentity=True)

print(f"input: {x_input}")
print(f"Network linear output: {out_with_linear}; network identity output: {out_with_identity}")
# is this not a dimensionality issue?

# Now that we have learned how to define layers and models in PyTorch using both the Sequential API and subclassing nn.Module, we're ready to turn our attention to how to actually implement network training with backpropagation.
#
# 1.4 Automatic Differentiation in PyTorch
# In PyTorch, torch.autograd is used for automatic differentiation, which is critical for training deep learning models with backpropagation.
#
# We will use the PyTorch .backward() method to trace operations for computing gradients. On a tensor, the requires_grad attribute controls whether autograd should record operations on that tensor. When a forward pass is made through the network, PyTorch builds a computational graph dynamically; then, to compute the gradient, the backward() method is called to perform backpropagation.
#
# Let's compute the gradient of  ...


### Gradient computation ###

# y = x^2
# Example: x = 3.0
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()  # Compute the gradient

dy_dx = x.grad
print("dy_dx of y=x^2 at x=3.0 is: ", dy_dx)
assert dy_dx == 6.0

# In training neural networks, we use differentiation and stochastic gradient descent (SGD) to optimize a loss function. Now that we have a sense of how PyTorch's autograd can be used to compute and access derivatives, we will look at an example where we use automatic differentiation and SGD to find the minimum of
# ...
#  is a variable for a desired value we are trying to optimize for;
# L
#  represents a loss that we are trying to minimize. While we can clearly solve this problem analytically (...), considering how we can compute this using PyTorch's autograd sets us up nicely for future labs where we use gradient descent to optimize entire neural network losses.


### Function minimization with autograd and gradient descent ###

# Initialize a random value for our initial x
x = torch.randn(1)
print(f"Initializing x={x.item()}")

learning_rate = 1e-2  # Learning rate
history = []
x_f = 4  # Target value


# We will run gradient descent for a number of iterations. At each iteration, we compute the loss,
#   compute the derivative of the loss with respect to x, and perform the update.
for i in range(500):
    x = torch.tensor([x], requires_grad=True)

    # TODO: Compute the loss as the square of the difference between x and x_f
    loss = (x-x_f) ** 2

    # Backpropagate through the loss to compute gradients
    loss.backward()

    # Update x with gradient descent
    x = x.item() - learning_rate * x.grad

    history.append(x.item())

# Plot the evolution of x as we optimize toward x_f!
plt.plot(history)
plt.plot([0, 500], [x_f, x_f])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.show()

# Now, we have covered the fundamental concepts of PyTorch -- tensors, operations, neural networks, and automatic differentiation. Fire!!