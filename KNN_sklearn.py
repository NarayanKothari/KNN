import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt 

data = pd.read_csv("Iris.csv") ### loading of data set
X = data.to_numpy() ## making X as the features or the input parameters  
X = X[:,1:5]
Y = data.to_numpy()## Making Y to contain the output(label) of the data points or features
Y = Y[:,5]


## Splitting the data such that 80% of the data used for training the model and
## 20% data is used to test the modes
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2) 


## Empty list which will hold the accuracy values of corresponding K values
test_accuracy = []
K = np.arange(1,35) 

# Loop over K values 
for k in K: 
    knn = KNeighborsClassifier(n_neighbors=k) ## creation of model
    knn.fit(X_train, y_train)  ## fitting of the model
    # Compute traning and test data accuracy 
    test_accuracy.append(knn.score(X_test, y_test)) ## predicting and the calculating the accuracy
  
# Generate plot 
plt.plot(K, test_accuracy, label = 'Testing dataset Accuracy') 
plt.legend() 
plt.xlabel('n_neighbors') 
plt.ylabel('Accuracy') 
plt.show() 