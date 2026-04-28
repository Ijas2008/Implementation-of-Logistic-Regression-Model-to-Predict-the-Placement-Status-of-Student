# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Initialize parameters**
   Set weights and bias to zero or small random values, and choose a learning rate.

2. **Apply sigmoid function for prediction**
   For each input (student features like marks, skills, etc.), compute the linear combination and pass it through the sigmoid function to get probability (placed = 1, not placed = 0).

3. **Compute error and update weights**
   Calculate the difference between actual and predicted values using a loss function (like log loss), then update weights using gradient descent.

4. **Repeat until convergence**
   Train the model over multiple iterations (epochs) until the prediction accuracy improves and loss is minimized.


## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data = pd.read_csv("Placement_Data (1).csv")
data = data.drop("salary", axis=1)
data = pd.get_dummies(data, drop_first=True)
X = data.drop("status_Placed", axis=1)
y = data["status_Placed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))
X1 = X.iloc[:, 0].values.reshape(-1, 1)
model_plot = LogisticRegression(max_iter=1000)
model_plot.fit(X1, y)
plt.scatter(X1, y, color='blue')
x_values = np.linspace(X1.min(), X1.max(), 100)
y_values = model_plot.predict_proba(x_values.reshape(-1,1))[:,1]
plt.plot(x_values, y_values)
plt.xlabel("Feature")
plt.ylabel("Probability")
plt.title("Logistic Regression Curve")
plt.show()

Developed by: IJAS J
RegisterNumber: 212225230102

```

## Output:
<img width="708" height="587" alt="image" src="https://github.com/user-attachments/assets/f5b3b467-b299-41df-b6c6-82708ad260ab" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
