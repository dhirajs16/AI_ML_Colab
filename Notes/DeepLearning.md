# **Deep Learning**

## Neural Network with an Analogy
    
  [But what is a neural network? | Deep learning chapter 1 - YouTube](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
  
  ### **Understanding Neurons and Neural Network with an analogy**
  
  **A neural network for classifying handwritten numbers form 0-9**
  
  - Consider each pixel of a 28x28 image of a handwritten number, as a **neuron. That make 784 neurons in total.**
      
      ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/25178473-2914-4d7b-a847-2a1cc1d5c641/bab2debe-6a42-46d0-9d8a-260e9bde84e3/image.png)
      
  - Each neuron holds a number between 0 and 1 (for example, 0.58). This number is called an **Activation**. Neurons with activation’s close to 0 appear black, while those close to 1 appear white.
  - These 784 neurons forms input layer of the neural network.
  
  [20241221-1514-23.6947215.mp4](https://prod-files-secure.s3.us-west-2.amazonaws.com/25178473-2914-4d7b-a847-2a1cc1d5c641/e23efef0-5964-4143-8c86-c96d33f64191/20241221-1514-23.6947215.mp4)
  
  - Each numbers are composed of edges and components (like a 9 is composed of a ring and vertical bar with a curve bottom).
      
      ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/25178473-2914-4d7b-a847-2a1cc1d5c641/5b2069a5-5b10-4a16-9c98-93452e1282f4/image.png)
      
  - Further these components are made of various edges.
        
      ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/25178473-2914-4d7b-a847-2a1cc1d5c641/fa9c423a-b879-4a16-ba84-9f9621281923/image.png)
      
  - The next layer after the input layer contains 16 neurons (an arbitrary choice). Each neuron corresponds to a specific edge, acting as a filter that activates when it detects a similar pattern. When a neuron "fires," its activation value approaches 1. So the activation of neurons of this layer is determined by previous layer neurons.
  - The second hidden layer is also composed of 16 neurons. Each neuron corresponds to a specific component—like a ring, vertical bar, or horizontal bar. Like the previous layer, it acts as a filter where similar neurons fire.

    
[Video 3](https://youtu.be/fne_UE7hDn0?si=8WsJVmuqCSi2Ps8r)

### **Neural Network**

Neural networks are a type of artificial intelligence that mimic the way the human brain processes information. They consist of interconnected groups of nodes, or artificial neurons, organized into layers. Here's a simple breakdown:

## **Basic Structure**

- **Input Layer**: This is where the neural network receives data. Each node in this layer corresponds to a feature of the input data.
- **Hidden Layers**: These layers process the information received from the input layer. The number of hidden layers can vary depending on the complexity of the task.
- **Output Layer**: This layer produces the final result or prediction based on the processed information.

## Types of Neural Networks

Neural networks are a cornerstone of modern artificial intelligence, enabling machines to learn from data. They can be categorized in various ways, primarily based on their architecture and the types of problems they solve. Here are some of the most prominent types of neural networks:

### **1. Feedforward Neural Networks (FNN)**

- **Description**: The simplest type of artificial neural network where connections between the nodes do not form cycles. Data moves in one direction—from input to output.
- **Applications**: Used in pattern recognition, regression tasks, and basic classification problems.

### **2. Convolutional Neural Networks (CNN)**

- **Description**: Specialized for processing grid-like data such as images. CNNs utilize convolutional layers to automatically learn spatial hierarchies from images.
- **Key Components**: Convolutional layers, pooling layers, and fully connected layers.
- **Applications**: Image classification, object detection, medical image analysis, and video processing[1][2][5].

### **3. Recurrent Neural Networks (RNN)**

- **Description**: Designed for sequential data processing, RNNs have loops that allow information to persist. They can use previous outputs as inputs for current tasks.
- **Challenges**: Issues like vanishing gradients can hinder performance on long sequences.
- **Applications**: Language modeling, speech recognition, and time series prediction[1][3][6].

### **4. Long Short-Term Memory Networks (LSTM)**

- **Description**: A type of RNN that includes memory cells to better capture long-term dependencies and mitigate the vanishing gradient problem.
- **Applications**: Language translation, speech recognition, and time-series forecasting[1][2].

### **5. Gated Recurrent Units (GRU)**

- **Description**: Similar to LSTMs but with a simpler architecture. GRUs use gating mechanisms to control the flow of information.
- **Advantages**: Fewer parameters compared to LSTMs while maintaining performance.
- **Applications**: Similar to LSTMs, including language modeling and sequence prediction[1][3].

### **6. Radial Basis Function Networks (RBFN)**

- **Description**: Uses radial basis functions as activation functions and is typically used for function approximation and pattern recognition.
- **Applications**: Classification tasks where the decision boundary is non-linear[3].

### **7. Autoencoders**

- **Description**: A type of feedforward neural network used primarily for unsupervised learning tasks. They consist of an encoder that compresses data and a decoder that reconstructs it.
- **Applications**: Dimensionality reduction, noise reduction, and feature learning.

### **8. Generative Adversarial Networks (GAN)**

- **Description**: Comprises two networks—a generator and a discriminator—that compete against each other. The generator creates fake data while the discriminator evaluates its authenticity.
- **Applications**: Image generation, style transfer, and data augmentation.

### **9. Deep Belief Networks (DBN)**

- **Description**: A stack of restricted Boltzmann machines that can learn to probabilistically reconstruct their inputs.
- **Applications**: Image recognition and feature extraction tasks[1][3].

### **10. Transformer Networks**

- **Description**: Utilizes self-attention mechanisms to process sequences in parallel rather than sequentially, making them efficient for large datasets.
- **Applications**: Natural language processing tasks such as translation and text summarization.

<img src="https://miro.medium.com/v2/resize:fit:2000/1*cuTSPlTq0a_327iTPJyD-Q.png" >

<hr>
## What is Perceptron?
    
[Video 4](https://www.youtube.com/watch?v=X7iIKPoZ0Sw&list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn&index=4)

A **perceptron** is a fundamental component of neural networks and a type of machine learning algorithm used for supervised learning tasks, particularly in binary classification problems. It is considered the simplest form of a neural network and serves as a building block for more complex neural networks.

### Key Components of a Perceptron

1. **Input Layer**: This layer receives the input data, which is typically represented as a vector of numerical values.
2. **Weights and Bias**: Each input is associated with a weight that determines its influence on the output. The bias acts as an intercept in a linear equation.
3. **Summation Function**: The perceptron calculates the weighted sum of its inputs by multiplying each input by its corresponding weight and adding these values together.
4. **Activation Function**: The weighted sum is then passed through an activation function, which determines the output of the perceptron. Common activation functions include the step function, sign function, and sigmoid function[1][2][3].

### How Perceptrons Work

1. **Training Process**: During training, the perceptron is presented with labeled examples. It compares its predicted output with the actual output and adjusts its weights to minimize the error[5].
2. **Binary Classification**: Perceptrons are primarily used for binary classification tasks, where they classify input data into one of two categories (e.g., 0 or 1)[4].

### Types of Perceptrons

- **Single-Layer Perceptron**: Limited to learning linearly separable patterns, it is effective for tasks where data can be divided into distinct categories using a straight line[4].
- **Multi-Layer Perceptron**: Consists of two or more layers, capable of handling more complex patterns and relationships within the data[4].
