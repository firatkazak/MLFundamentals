from sklearn.cross_decomposition import PLSRegression
import numpy as np

# Data: Experience (years), Education level (years) and Satisfaction level (1-5)
X = np.array([[1, 10], [2, 11], [3, 12], [4, 13], [5, 14], [6, 15]])
y = np.array([[1, 30000], [2, 35000], [3, 40000], [4, 45000], [5, 50000], [6, 60000]])

# Create PLS regression model
model = PLSRegression(n_components=2)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

print("Predicted Satisfaction Levels and Salaries: ", y_pred)

# Partial Least Squares (PLS) Regression
# PLS regression, birden fazla bağımlı ve bağımsız değişken arasında ilişkiyi modellemek için kullanılır.
