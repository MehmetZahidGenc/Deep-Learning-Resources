# Vanishing Gradient Problem

The primary reason we use the ResNet structure is that we encountered this problem during the training.

==> As more layers using certain activation functions are added to neural networks, the gradients of the loss function approaches zero, making the network hard to train.

When 'n' hidden layers use an activation function like the sigmoid function, n small derivatives are multiplied together. Thus, the gradient decreaes exponentially       as we propagate down to initial layers. 
    
A small gradient means that the weight and biases of the initial layers will not be updated effectively with each training session. Since these initial layers are       often cruical to recognizing the core element of the input data, it can lead to overall inaccuracy of the whole network.
    
    
# Solution

1) The simplest solution is to use other activation functions, such as ReLU, which does not cause a small derivative
2) Residual networks are another solution, as they provide residual connections straight to earlier layers

![image](https://user-images.githubusercontent.com/71969819/179409152-3f845fc6-d1cc-40b1-8981-1f39e4d830cd.png)


# What is Batch normalization ?

Batch normalization (also known as batch norm) is a method used to make training of artificial neural networks faster and more stable through normalization of the layers' inputs by re-centering and re-scaling.
