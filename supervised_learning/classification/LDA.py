
import numpy as np 
from sklearn.discriminant_analysis 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.model_selection import train_test_split 

# Generate random data 
X = np.random.randn(100, 10) 
y = np.random.randint(2, size=100) 

# Split the data into training and test sets 
X_train, X_test,\ 
	y_train, y_test = train_test_split(X, y, 
									test_size=0.3) 

# Create a LinearDiscriminantAnalysis estimator 
# and fit it to the training data 
estimator = LinearDiscriminantAnalysis(shrinkage=None) 
estimator.fit(X_train, y_train) 

# Obtain predictions for the test set 
y_pred = estimator.predict(X_test) 

# Print the classification accuracy 
print(estimator.score(X_test, y_test)) 

