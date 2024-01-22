The main motive of this project is to anticipate whether an IoT networked device is under attack or not with the help of machine learning. To attain the objectives, we present a comprehensive study on the application of machine learning (ML) and deep learning (DL) techniques for Intrusion Detection Systems (IDS) in IoT networks. Leveraging the CIC IoT 2023 dataset from the Canadian Institution of Cybersecurity, we explore three distinct classification scenarios: 34-class, 8-class, and binary. Employing feature selection to retain the mostsignificant features, we evaluate ten machine learning models(KNN, logistic regression, decision tree, random forest, gradientboost, xgboost, adaboost, light gbm, catboost, and naive bayes) alongside three deep learning techniques (ANN, LSTM, GRU).

->Dataset Preprocessing & Feature Selection
-Handle Missing Data Columns
-Drop Duplicate Columns
-Handle Outliers
-Label Encoding using Label Encoder
-Feature Selection using Random Forest Classifier
-Feature Scaling using Standard Scaler

->Classifications:
-34-class Classification
-8-class Classification
-binary Classification

->Machine Learning Models: This subsection provides a
brief overview of the machine learning models employed in
our study:
• Naive Bayes: A probabilistic model based on Bayes’
theorem, particularly effective for text classification tasks.
• Logistic Regression: A regression analysis that models
the probability of a binary outcome, often used for
classification problems.
• Decision Tree: A tree-like model that makes decisions
based on feature values, suitable for hierarchical classification.
• Random Forest: An ensemble of decision trees that improves accuracy and minimizes overfitting.
• K-Nearest Neighbors (KNN): A non-parametric method
classifying data points based on their proximity to other
points.
• Gradient Boost: An ensemble technique that builds decision trees sequentially, correcting errors of the preceding
trees.
• XGBoost: An optimized gradient boosting library that
enhances speed and performance.
• Adaboost: A boosting algorithm that combines weak
learners to create a strong classifier.
• Light GBM: A gradient boosting framework designed for
distributed and efficient training.
• CatBoost: A boosting algorithm that handles categorical
features efficiently.

->Deep Learning Models: This subsection presents a brief
overview of the deep learning models utilized:
• Artificial Neural Networks (ANN): A model inspired by
the human brain’s neural structure, effective for complex
pattern recognition. For all three types of classifications
the ANN model consists of a simple feedforward
neural network with two hidden layers, each having 64
neurons and using ReLU activation. The output layer
has 34 neurons with softmax activation for a multi-class
classification task. The model is compiled using the
Adam optimizer, sparse categorical crossentropy loss,
and accuracy as the evaluation metric.
• Long Short-Term Memory (LSTM): A type of recurrent
neural network (RNN) designed to capture long-term
dependencies in sequential data. This DL model is a
stacked LSTM architecture with batch normalization
and dropout layers. The model is compiled using the
Adam optimizer, sparse categorical crossentropy loss,
and accuracy as the evaluation metric. The combination
of LSTM layers, batch normalization, and dropout
contributes to the model’s ability to capture sequential
dependencies, generalize well, and prevent overfitting
during training. For 34-class, 8-class and binary
classification, the final LSTM layer has 34 units, 8 units
and 1 unit respectively, indicating the number of classes
in the output layer. The use of MAE loss suggests a
regression-style objective for the binary classification.
• Gated Recurrent Unit (GRU): Similar to LSTM, GRU is
a type of RNN designed for improved efficiency in capturing sequential patterns. This section we implemented
a GRU-based neural network architecture for sequence
data, particularly suitable for time-series analysis. The
model includes multiple GRU layers with batch normalization and dropout for regularization. The Adam
optimizer is employed for training, and the model is
compiled with appropriate loss and evaluation metrics
for the specific task at hand. For 34-class, 8-class and
binary classification, the final LSTM layer has 34 units,
8 units and 1 unit respectively, indicating the number of
classes in the output layer. The use of MAE loss suggests
a regression-style objective for the binary classification
and emphasizes accuracy during training.





