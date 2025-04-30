
# Laboratory 2: Computer Vision
# Part 1: MNIST Digit Classification
# In the first portion of this lab, we will build and train a convolutional neural network (CNN) for classification of handwritten digits from the famous MNIST dataset. The MNIST dataset consists of 60,000 training images and 10,000 test images. Our classes are the digits 0-9.
#
# First, let's download the course repository, install dependencies, and import the relevant packages we'll need for this lab.

# Import PyTorch and other relevant libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary


import mitdeeplearning as mdl

# other packages
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

assert torch.cuda.is_available(), "Please enable GPU from runtime settings"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1.1 MNIST dataset
# Let's download and load the dataset and display a few random samples from it:


# Download and transform the MNIST dataset
transform = transforms.Compose([
    # Convert images to PyTorch tensors which also scales data from [0,255] to [0,1]
    transforms.ToTensor()
])

# Download training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# The MNIST dataset object in PyTorch is not a simple tensor or array. It's an iterable dataset that loads samples (image-label pairs) one at a time or in batches. In a later section of this lab, we will define a handy DataLoader to process the data in batches.

image, label = train_dataset[0]
print(image.size())  # For a tensor: torch.Size([1, 28, 28])
print(label)  # For a label: integer (e.g., 5)


# Evaluate accuracy on the test dataset
# Now that we've trained the model, we can ask it to make predictions about a test set that it hasn't seen before. In this example, iterating over the testset_loader allows us to access our test images and test labels. And to evaluate accuracy, we can check to see if the model's predictions match the labels from this loader.
#
# Since we have now trained the mode, we will use the eval state of the model on the test dataset.

'''TODO: Use the model we have defined in its eval state to complete
and call the evaluate function, and calculate the accuracy of the model'''

def evaluate(model, dataloader, loss_function):
    # Evaluate model performance on the test dataset
    model.eval()
    test_loss = 0
    correct_pred = 0
    total_pred = 0
    # Disable gradient calculations when in inference mode
    # WHY?
    with torch.no_grad():
        for images, labels in testset_loader:
            # ensure evaluation happens on the GPU
            images, labels = images.to(device), labels.to(device)

            # feed the images into the model and obtain the predictions (forward pass)
            outputs = model(images)

            loss = loss_function(outputs, labels)

            # TODO: Calculate test loss
            test_loss +=  loss.item() * images.size(0)

            # make a prediction and determine whether it is correct!
            # TODO: identify the digit with the highest probability prediction for the images in the test dataset.
            predicted = torch.argmax(outputs, dim=1)

            # TODO: tally the number of correct predictions
            correct_pred += (predicted == labels).sum().item()

            # TODO: tally the total number of predictions
            total_pred += labels.size(0)

    # Compute average loss and accuracy
    test_loss /= total_pred
    test_acc = correct_pred / total_pred
    return test_loss, test_acc

# You may observe that the accuracy on the test dataset is a little lower than the accuracy on the training dataset. This gap between training accuracy and test accuracy is an example of overfitting, when a machine learning model performs worse on new data than on its training data.
#
# What is the highest accuracy you can achieve with this first fully connected model? Since the handwritten digit classification task is pretty straightforward, you may be wondering how we can do better...


# Define the CNN model
# We'll use the same training and test datasets as before, and proceed similarly as our fully connected network to define and train our new CNN model. To do this we will explore two layers we have not encountered before: you can use nn.Conv2d to define convolutional layers and nn.MaxPool2D to define the pooling layers. Use the parameters shown in the network architecture above to define these layers and build the CNN model. You can decide to use nn.Sequential or to subclass nn.Modulebased on your preference.

### Basic CNN in PyTorch ###

# Instantiate the model
cnn_model = nn.Sequential(
nn.Conv2d(1, 32, kernel_size=3),
nn.ReLU(),
# nn.Sigmoid(),
nn.MaxPool2d(kernel_size=2),

nn.Conv2d(32, 64, kernel_size=3),
nn.ReLU(),
# nn.Sigmoid(),
nn.MaxPool2d(kernel_size=2),

nn.Flatten(),
nn.Linear(64 * 5 * 5, 128),
nn.ReLU(),
#nn.Sigmoid(), suprisinglky bad results

nn.Linear(128, 10)
    # torch.nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3),# in_channels=3 ,# filter_size= 3),
    # torch.nn.ReLU(),
    # torch.nn.MaxPool2d(kernel_size=2,stride=2),
    #
    # torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),# filter_size= 3),
    # torch.nn.ReLU(),
    # torch.nn.MaxPool2d(kernel_size=2,stride=2),
    #
    # torch.nn.Flatten(),
    # torch.nn.Linear(64*6*6, 1024),
    # torch.nn.ReLU(),
    # torch.nn.Linear(1024,10)

).to(device)

# Initialize the model by passing some data through
image, label = train_dataset[0]
image = image.to(device).unsqueeze(0)  # Add batch dimension → Shape: (1, 1, 28, 28)
output = cnn_model(image)
# Print the model summary
print(cnn_model)

# Train and test the CNN model
# Earlier in the lab, we defined a train function. The body of the function is quite useful because it allows us to have control over the training model, and to record differentiation operations during training by computing the gradients using loss.backward(). You may recall seeing this in Lab 1 Part 1.
#
# We'll use this same framework to train our cnn_model using stochastic gradient descent. You are free to implement the following parts with or without the train and evaluate functions we defined above. What is most important is understanding how to manipulate the bodies of those functions to train and test models.
#
# As we've done above, we can define the loss function, optimizer, and calculate the accuracy of the model. Define an optimizer and learning rate of choice. Feel free to modify as you see fit to optimize your model's performance.



# ...
#  what's the kernal


# Rebuild the CNN model

# Define hyperparams
batch_size = 64
epochs = 7
optimizer = optim.SGD(cnn_model.parameters(), lr=1e-2)

# TODO: instantiate the cross entropy loss function
loss_function = nn.CrossEntropyLoss()

# Redefine trainloader with new batch size parameter (tweak as see fit if optimizing)
trainset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
loss_history = mdl.util.LossHistory(smoothing_factor=0.95) # to record the evolution of the loss
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss', scale='semilogy')


if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

# Training loop!
cnn_model.train()

for epoch in range(epochs):
    total_loss = 0
    correct_pred = 0
    total_pred = 0

    # First grab a batch of training data which our data loader returns as a tensor
    for idx, (images, labels) in enumerate(tqdm(trainset_loader)):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        #'''TODO: feed the images into the model and obtain the predictions'''
        logits = cnn_model(images)
        # logits = # TODO

        #'''TODO: compute the categorical cross entropy loss
        loss = loss_function(logits, labels)
        # loss = # TODO
        # Get the loss and log it to comet and the loss_history record
        loss_value = loss.item()
        loss_history.append(loss_value) # append the loss to the loss_history record
        plotter.plot(loss_history.get())

        # Backpropagation/backward pass
        '''TODO: Compute gradients for all model parameters and propagate backwads
            to update model parameters. remember to reset your optimizer!'''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Get the prediction and tally metrics
        predicted = torch.argmax(logits, dim=1)
        correct_pred += (predicted == labels).sum().item()
        total_pred += labels.size(0)

    # Compute metrics
    total_epoch_loss = total_loss / total_pred
    epoch_accuracy = correct_pred / total_pred
    # print(f"Epoch {epoch + 1}, Loss: {total_epoch_loss}, Accuracy: {epoch_accuracy:.4f}")

plt.show()

# Evaluate the CNN Model
# Now that we've trained the model, let's evaluate it on the test dataset.

'''TODO: Evaluate the CNN model!'''

test_loss, test_acc = evaluate(cnn_model, trainset_loader, loss_function)
# test_loss, test_acc = # TODO

print('Test accuracy:', test_acc)

# What is the highest accuracy you're able to achieve using the CNN model, and how does the accuracy of the CNN model compare to the accuracy of the simple fully connected network? What optimizers and learning rates seem to be optimal for training the CNN model?

# Make predictions with the CNN model
# With the model trained, we can use it to make predictions about some images.

test_image, test_label = test_dataset[0]


test_image = test_image.to(device).unsqueeze(0)

# put the model in evaluation (inference) mode
cnn_model.eval()
predictions_test_image = cnn_model(test_image)

# With this function call, the model has predicted the label of the first image in the testing set. Let's take a look at the prediction:

print(predictions_test_image)

# As you can see, a prediction is an array of 10 numbers. Recall that the output of our model is a distribution over the 10 digit classes. Thus, these numbers describe the model's predicted likelihood that the image corresponds to each of the 10 different digits.
#
# Let's look at the digit that has the highest likelihood for the first image in the test dataset:

'''TODO: identify the digit with the highest likelihood prediction for the first
    image in the test dataset. '''
predictions_value = predictions_test_image.cpu().detach().numpy() #.cpu() to copy tensor to memory first
prediction = np.argmax(predictions_value)
# prediction = # TODO
print(prediction)

# So, the model is most confident that this image is a "???". We can check the test label (remember, this is the true identity of the digit) to see if this prediction is correct:

# print("Label of this digit is:", test_label)
# plt.imshow(test_image[0,0,:,:].cpu(), cmap=plt.cm.binary)
# plt.show()

# It is! Let's visualize the classification results on the MNIST dataset. We will plot images from the test dataset along with their predicted label, as well as a histogram that provides the prediction probabilities for each of the digits.
#
# Recall that in PyTorch the MNIST dataset is typically accessed using a DataLoader to iterate through the test set in smaller, manageable batches. By appending the predictions, test labels, and test images from each batch, we will first gradually accumulate all the data needed for visualization into singular variables to observe our model's predictions.


# Initialize variables to store all data
all_predictions = []
all_labels = []
all_images = []

# Process test set in batches
with torch.no_grad():
    for images, labels in testset_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = cnn_model(images)

        # Apply softmax to get probabilities from the predicted logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get predicted classes
        predicted = torch.argmax(probabilities, dim=1)

        all_predictions.append(probabilities)
        all_labels.append(labels)
        all_images.append(images)

all_predictions = torch.cat(all_predictions)  # Shape: (total_samples, num_classes)
all_labels = torch.cat(all_labels)            # Shape: (total_samples,)
all_images = torch.cat(all_images)            # Shape: (total_samples, 1, 28, 28)

# Convert tensors to NumPy for compatibility with plotting functions
predictions = all_predictions.cpu().numpy()  # Shape: (total_samples, num_classes)
test_labels = all_labels.cpu().numpy()       # Shape: (total_samples,)
test_images = all_images.cpu().numpy()       # Shape: (total_samples, 1, 28, 28)


# We can also plot several images along with their predictions, where correct prediction labels are blue and incorrect prediction labels are grey. The number gives the percent confidence (out of 100) for the predicted label. Note the model can be very confident in an incorrect prediction!

# Plots the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 10
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  mdl.lab2.plot_image_prediction(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  mdl.lab2.plot_value_prediction(i, predictions, test_labels)
plt.show()

# with torch.no_grad():


# us optimize to see what it thinks digits look like

inputTensor = torch.rand(1, 1, 28, 28).to(device)
inputTensor.requires_grad = True

# plt.imshow(inputTensor.cpu(), cmap=plt.cm.binary)
# plt.show()

optimizer = torch.optim.Adam([inputTensor], lr=.01)


for i in range(100000):

    logits = cnn_model(inputTensor)
    if i %1000 ==0:
        print(logits)
    # '''TODO: compute the categorical cross entropy loss
    loss = loss_function(logits, torch.tensor([9]).to(device))
    # torch.optim.Adam(inputTensor, lr=learning_rate)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.imshow(inputTensor[0,0,:,:].cpu().detach().numpy(), cmap=plt.cm.binary)
plt.show()
