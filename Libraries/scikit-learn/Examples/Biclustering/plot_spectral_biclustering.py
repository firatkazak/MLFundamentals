from matplotlib import pyplot as plt
from sklearn.datasets import make_checkerboard
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score
import numpy as np
import os

os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# Spectral Biclustering algoritması Örneği;

n_clusters = (4, 3)
data, rows, columns = make_checkerboard(shape=(300, 300),
                                        n_clusters=n_clusters,
                                        noise=10,
                                        shuffle=False,
                                        random_state=42
                                        )

plt.matshow(A=data,
            cmap=plt.cm.Blues
            )

plt.title("Original dataset")
plt.show()

# Creating lists of shuffled row and column indices
rng = np.random.RandomState(0)
row_idx_shuffled = rng.permutation(data.shape[0])
col_idx_shuffled = rng.permutation(data.shape[1])

# We redefine the shuffled data and plot it. We observe that we lost the structure of original data matrix.
data = data[row_idx_shuffled][:, col_idx_shuffled]

plt.matshow(A=data,
            cmap=plt.cm.Blues
            )

plt.title(label="Shuffled dataset")
plt.show()

#
model = SpectralBiclustering(n_clusters=n_clusters,
                             method="log",
                             random_state=0
                             )

model.fit(data)

# Compute the similarity of two sets of biclusters
score = consensus_score(a=model.biclusters_,
                        b=(rows[:, row_idx_shuffled], columns[:, col_idx_shuffled])
                        )

print(f"consensus score: {score:.1f}")

# Reordering first the rows and then the columns.
reordered_rows = data[np.argsort(a=model.row_labels_)]
reordered_data = reordered_rows[:, np.argsort(a=model.column_labels_)]

plt.matshow(A=reordered_data,
            cmap=plt.cm.Blues
            )

plt.title(label="After biclustering; rearranged to show biclusters")
plt.show()

plt.matshow(A=np.outer(np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) + 1),
            cmap=plt.cm.Blues
            )

plt.title("Checkerboard structure of rearranged data")
plt.show()
