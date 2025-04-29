import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
np.random.seed(0)
x=2*np.random.rand(100,1)
y=4+3*x + np.random.rand(100,1)
X_train, X_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)
model= LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plt.scatter(X_test, y_test, color='b', label='Actual Data')
plt.plot(X_test, y_pred, color='r', label='Regression Line')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('Linear Regression')
plt.show()
