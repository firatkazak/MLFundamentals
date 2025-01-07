import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

# Probability Calibration for 3-class classification Örneği;

np.random.seed(0)

X, y = make_blobs(n_samples=2000,
                  n_features=2,
                  centers=3,
                  random_state=42,
                  cluster_std=5.0
                  )

X_train, y_train = X[:600], y[:600]
X_valid, y_valid = X[600:1000], y[600:1000]
X_train_valid, y_train_valid = X[:1000], y[:1000]
X_test, y_test = X[1000:], y[1000:]

clf = RandomForestClassifier(n_estimators=25)

clf.fit(X=X_train_valid,
        y=y_train_valid
        )

clf = RandomForestClassifier(n_estimators=25)

clf.fit(X=X_train,
        y=y_train
        )

cal_clf = CalibratedClassifierCV(estimator=FrozenEstimator(clf),
                                 method="sigmoid"
                                 )

cal_clf.fit(X_valid, y_valid)

plt.figure(figsize=(10, 10))
colors = ["r", "g", "b"]

clf_probs = clf.predict_proba(X_test)
cal_clf_probs = cal_clf.predict_proba(X_test)

# Plot arrows
for i in range(clf_probs.shape[0]):
    plt.arrow(
        x=clf_probs[i, 0],
        y=clf_probs[i, 1],
        dx=cal_clf_probs[i, 0] - clf_probs[i, 0],
        dy=cal_clf_probs[i, 1] - clf_probs[i, 1],
        color=colors[y_test[i]],
        head_width=1e-2,
    )

# Plot perfect predictions, at each vertex
plt.plot([1.0], [0.0], "ro", ms=20, label="Class 1")
plt.plot([0.0], [1.0], "go", ms=20, label="Class 2")
plt.plot([0.0], [0.0], "bo", ms=20, label="Class 3")

# Plot boundaries of unit simplex
plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], "k", label="Simplex")

# Annotate points 6 points around the simplex, and mid point inside simplex
plt.annotate(text=r"($\frac{1}{3}$, $\frac{1}{3}$, $\frac{1}{3}$)",
             xy=(1.0 / 3, 1.0 / 3),
             xytext=(1.0 / 3, 0.23),
             xycoords="data",
             arrowprops=dict(facecolor="black", shrink=0.05),
             horizontalalignment="center",
             verticalalignment="center",
             )

plt.plot([1.0 / 3], [1.0 / 3], "ko", ms=5)

plt.annotate(text=r"($\frac{1}{2}$, $0$, $\frac{1}{2}$)",
             xy=(0.5, 0.0),
             xytext=(0.5, 0.1),
             xycoords="data",
             arrowprops=dict(facecolor="black", shrink=0.05),
             horizontalalignment="center",
             verticalalignment="center",
             )

plt.annotate(text=r"($0$, $\frac{1}{2}$, $\frac{1}{2}$)",
             xy=(0.0, 0.5),
             xytext=(0.1, 0.5),
             xycoords="data",
             arrowprops=dict(facecolor="black", shrink=0.05),
             horizontalalignment="center",
             verticalalignment="center",
             )

plt.annotate(text=r"($\frac{1}{2}$, $\frac{1}{2}$, $0$)",
             xy=(0.5, 0.5),
             xytext=(0.6, 0.6),
             xycoords="data",
             arrowprops=dict(facecolor="black", shrink=0.05),
             horizontalalignment="center",
             verticalalignment="center",
             )

plt.annotate(text=r"($0$, $0$, $1$)",
             xy=(0, 0),
             xytext=(0.1, 0.1),
             xycoords="data",
             arrowprops=dict(facecolor="black", shrink=0.05),
             horizontalalignment="center",
             verticalalignment="center",
             )

plt.annotate(text=r"($1$, $0$, $0$)",
             xy=(1, 0),
             xytext=(1, 0.1),
             xycoords="data",
             arrowprops=dict(facecolor="black", shrink=0.05),
             horizontalalignment="center",
             verticalalignment="center",
             )

plt.annotate(text=r"($0$, $1$, $0$)",
             xy=(0, 1),
             xytext=(0.1, 1),
             xycoords="data",
             arrowprops=dict(facecolor="black", shrink=0.05),
             horizontalalignment="center",
             verticalalignment="center",
             )

# Add grid
plt.grid(False)

for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    plt.plot([0, x], [x, 0], "k", alpha=0.2)
    plt.plot([0, 0 + (1 - x) / 2], [x, x + (1 - x) / 2], "k", alpha=0.2)
    plt.plot([x, x + (1 - x) / 2], [0, 0 + (1 - x) / 2], "k", alpha=0.2)

plt.title("Change of predicted probabilities on test samples after sigmoid calibration")
plt.xlabel("Probability class 1")
plt.ylabel("Probability class 2")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
_ = plt.legend(loc="best")

score = log_loss(y_test, clf_probs)
cal_score = log_loss(y_test, cal_clf_probs)

print("Log-loss of")
print(f" * uncalibrated classifier: {score:.3f}")
print(f" * calibrated classifier: {cal_score:.3f}")

plt.figure(figsize=(10, 10))

# Generate grid of probability values
p1d = np.linspace(start=0, stop=1, num=20)
p0, p1 = np.meshgrid(p1d, p1d)
p2 = 1 - p0 - p1
p = np.c_[p0.ravel(), p1.ravel(), p2.ravel()]
p = p[p[:, 2] >= 0]

# Use the three class-wise calibrators to compute calibrated probabilities
calibrated_classifier = cal_clf.calibrated_classifiers_[0]
prediction = np.vstack(
    [
        calibrator.predict(this_p)
        for calibrator, this_p in zip(calibrated_classifier.calibrators, p.T)
    ]
).T

prediction /= prediction.sum(axis=1)[:, None]

# Plot changes in predicted probabilities induced by the calibrators
for i in range(prediction.shape[0]):
    plt.arrow(x=p[i, 0],
              y=p[i, 1],
              dx=prediction[i, 0] - p[i, 0],
              dy=prediction[i, 1] - p[i, 1],
              head_width=1e-2,
              color=colors[np.argmax(p[i])],
              )

# Plot the boundaries of the unit simplex
plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], "k", label="Simplex")

plt.grid(False)

for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    plt.plot([0, x], [x, 0], "k", alpha=0.2)
    plt.plot([0, 0 + (1 - x) / 2], [x, x + (1 - x) / 2], "k", alpha=0.2)
    plt.plot([x, x + (1 - x) / 2], [0, 0 + (1 - x) / 2], "k", alpha=0.2)

plt.title("Learned sigmoid calibration map")
plt.xlabel("Probability class 1")
plt.ylabel("Probability class 2")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

plt.show()
