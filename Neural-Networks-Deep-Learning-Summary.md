# **Neural Networks & Deep Learning Summary**

## **1️⃣ What is a Neural Network?**  
A **neural network** is a system of layers of neurons that process input data and make predictions.  
It consists of:  
- **Input layer** → Takes input features.  
- **Hidden layers** → Process data through weights, biases, and activation functions.  
- **Output layer** → Produces the final result.  

✅ **At the core, neural networks are just matrix multiplications followed by activation functions!**  

---

## **2️⃣ Understanding Neural Network Parameters**  
To calculate the **total parameters (weights + biases)**, use the formula:  

\[ \sum_{l=1}^{n+1} \left( N_{l-1} \times N_l + N_l \right) \]

Where:  
- \( N_l \) = Number of neurons in layer \( l \)  
- \( N_{l-1} \) = Number of neurons in the previous layer  
- **Weights** = \( N_{l-1} \times N_l \)  
- **Biases** = \( N_l \)  

✅ **Example**: If we have a network with:  
- **4 input neurons**  
- **Hidden Layer 1 (3 neurons)**  
- **Hidden Layer 2 (3 neurons)**  
- **Hidden Layer 3 (4 neurons)**  
- **Output layer (2 neurons)**  

The **total parameters** = **53** (we calculated this earlier).  

---
## **3️⃣ Activation Functions**  
Activation functions determine how neurons process input signals. Some commonly used ones are:

### **ReLU (Rectified Linear Unit)**  
\[ f(x) = \max(0, x) \]  
- Used in hidden layers.  
- Keeps positive values as-is, sets negative values to zero.  
- Helps prevent vanishing gradient problems.

### **Sigmoid**  
\[ f(x) = \frac{1}{1 + e^{-x}} \]  
- Used in binary classification problems.  
- Outputs values between 0 and 1.

### **Softmax**  
\[ f(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}} \]  
- Used for multi-class classification.  
- Converts outputs into probabilities by ensuring they sum up to 1.

---

## **3️⃣ The Complete Deep Learning Process**  
Neural networks are not just about defining layers. **Deep learning involves a full pipeline:**  

### **✅ 1. Data Collection & Preprocessing**  
- Gather dataset, clean data, normalize features, split into training/testing sets.

### **✅ 2. Define the Neural Network Architecture**  
- Choose the number of layers and neurons.  
- Decide activation functions (ReLU, Sigmoid, Softmax).  
- Initialize weights and biases.  

### **✅ 3. Choose a Loss Function & Optimizer**  
- Loss function → Measures error (e.g., MSE, Cross-Entropy).  
- Optimizer → Adjusts weights (e.g., Adam, SGD).  

### **✅ 4. Training (Forward + Backpropagation)**  
- **Forward propagation** → Compute outputs.  
- **Loss calculation** → Compare predicted vs actual.  
- **Backpropagation** → Adjust weights using gradients.  
- **Repeat for multiple epochs** until the loss is minimized.  

✅ **Forward + Backpropagation are core parts of neural networks!**  

### **✅ 5. Model Evaluation**  
- Test on unseen data.  
- Use metrics like accuracy, RMSE, precision, recall.  

### **✅ 6. Make Predictions & Deployment**  
- The trained model is used for real-world applications.  
- Deploy in a web app, API, or edge device.  

---

## **4️⃣ Sample Example: Neural Network Computation**
Let's take an example to better understand how weights, biases, and activations work.

### **Problem Statement**
We want to predict a student's performance based on **study hours (X1)** and **sleep hours (X2)** using a simple neural network.

### **Given Data (Input Features)**
| Study Hours (X1) | Sleep Hours (X2) | Performance (Y - Target) |
|-----------------|-----------------|--------------------------|
| 2              | 8               | 0.6                      |
| 5              | 6               | 0.8                      |

### **Neural Network Architecture**
- **Input Layer (2 neurons):** X1 and X2.
- **Hidden Layer (2 neurons):** Activation function = ReLU.
- **Output Layer (1 neuron):** Activation function = Sigmoid (since we need a probability output between 0 and 1).

### **Step 1: Initialize Weights and Biases**
Let’s assume random values:
- **Weights for Hidden Layer:**  
  
  \[ W_1 = \begin{bmatrix} 0.1 & 0.4 \\ 0.2 & 0.3 \end{bmatrix} \]
  
- **Biases for Hidden Layer:**  
  
  \[ B_1 = \begin{bmatrix} 0.5 \\ 0.6 \end{bmatrix} \]
  
### **Step 2: Forward Propagation**
#### **Hidden Layer Calculation**
\[ H = ReLU(W_1 \times X + B_1) \]
Substituting values:
\[ H = ReLU(\begin{bmatrix} 0.1 & 0.4 \\ 0.2 & 0.3 \end{bmatrix} \times \begin{bmatrix} 2 \\ 8 \end{bmatrix} + \begin{bmatrix} 0.5 \\ 0.6 \end{bmatrix}) \]

\[ H = ReLU(\begin{bmatrix} 3.8 \\ 3.2 \end{bmatrix}) \]
Since ReLU keeps positive values as-is:
\[ H = \begin{bmatrix} 3.8 \\ 3.2 \end{bmatrix} \]

### **Step 3: Python Implementation**
```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Inputs
X = np.array([[2], [8]])

# Weights and biases for hidden layer
W1 = np.array([[0.1, 0.4], [0.2, 0.3]])
B1 = np.array([[0.5], [0.6]])

# Forward pass - Hidden Layer
H = relu(np.dot(W1, X) + B1)

# Weights and bias for output layer
W2 = np.array([[0.7, 0.9]])
B2 = np.array([[0.2]])

# Forward pass - Output Layer
O = sigmoid(np.dot(W2, H) + B2)

print("Predicted Output:", O)
```

### **Step 4: Interpretation**
The output is **a probability value** that predicts student performance.  

---

## **5️⃣ Key Takeaways**  
✔ **Neural networks are based on matrix multiplications + activation functions.**  
✔ **The number of trainable parameters follows a formula.**  
✔ **Training = Forward Propagation + Backpropagation.**  
✔ **Deep learning is a full process, not just defining a network.**  
✔ **A real example helps in understanding how weights, biases, and activations work.**  

---

 
