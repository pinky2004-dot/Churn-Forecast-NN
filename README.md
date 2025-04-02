# 🏦 Bank Churn Prediction using Neural Networks

## 📌 Objective

In the banking industry, customer churn (customers leaving for competitors) is a critical challenge. To address this, this project builds a neural network-based classifier to predict whether a customer will leave the bank within the next six months. By identifying at-risk customers, banks can take proactive steps to improve retention strategies and customer satisfaction.

## 🛠️ Techniques & Topics Covered

This project applies a comprehensive set of machine learning and deep learning techniques, including:

### 1️⃣ Data Handling & Preprocessing
- Pandas & NumPy – Data manipulation and preprocessing
- Exploratory Data Analysis (EDA) – Understanding patterns and trends
- Data Cleaning – Handling missing values, outliers, and inconsistencies
- Inferential Statistics – Extracting meaningful insights from data
- Correlation Analysis – Identifying key factors driving customer churn

### 2️⃣ Feature Engineering & Handling Imbalance
- One-Hot Encoding – Transforming categorical variables for neural networks
- Standardization & Feature Scaling – Ensuring optimal model performance
- SMOTE (Synthetic Minority Over-sampling Technique) – Addressing class imbalance by generating synthetic data points

### 3️⃣ Model Development: Feedforward Neural Network (FNN)
- TensorFlow & Keras – Building and training deep learning models
- Sequential Model Architecture – Designing a fully connected neural network
- Activation Functions – Using ReLU, Sigmoid, and Softmax for different layers
- Loss Function & Optimizer –
  - SGD Optimizer (initial experiment)
  - Adam Optimizer (final model selection for improved performance and convergence)
- Hidden Layer Tuning – Experimenting with the number of neurons for optimal accuracy

### 4️⃣ Model Optimization & Performance Enhancements
- Dropout Regularization – Preventing overfitting by randomly deactivating neurons during training
- Batch Normalization – Stabilizing learning and improving convergence
- L1 & L2 Regularization – Controlling model complexity
- Hyperparameter Tuning – Finding the best combination of layers, neurons, and dropout rates

## 📊 Model Evaluation & Selection

### Metrics Considered

Since churn prediction requires minimizing false negatives (i.e., customers who actually leave but are misclassified as staying), the best evaluation metric is:

  ✅ Recall – Prioritizing recall ensures we capture as many churners as possible. A high recall score means fewer customers will be incorrectly classified as non-churners.

Business Translation of Model Outputs:
- True Positives (TP) – Correctly predicted churners
- False Negatives (FN) – Actual churners incorrectly classified as non-churners (critical to minimize)
- False Positives (FP) – Customers flagged as churners but who actually stay

Since false negatives result in lost revenue and missed retention opportunities, recall is the key metric to optimize.

### Performance Comparison

I evaluated multiple model configurations by adjusting optimizers, hidden layers, and handling class imbalance with SMOTE, dropout, and batch normalization, etc.

### Final Model Selection

❌ Models 12 - 14 were discarded due to clear overfitting issues.

✅ Models 1, 3, and 9 performed well, with Model 9 being the best choice.

- Model 9 had the best recall (0.858) while maintaining high precision and accuracy.
- Slight overfitting was present, but the model generalized well to unseen data.

## 📌 Practical Impact

This model is business-critical for banks as it:
- Identifies customers likely to leave, allowing proactive engagement strategies.
- Minimizes false negatives, ensuring retention efforts are directed at the right customers.
- Helps optimize marketing and loyalty programs, improving customer lifetime value.





