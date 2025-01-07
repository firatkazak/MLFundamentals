from matplotlib import pyplot as plt
from sklearn.cluster import SpectralCoclustering
from sklearn.datasets import make_biclusters
from sklearn.metrics import consensus_score
import numpy as np
import os

os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# Spectral Co-Clustering algoritması Örneği;

data, rows, columns = make_biclusters(shape=(300, 300),
                                      n_clusters=5,
                                      noise=5,
                                      shuffle=False,
                                      random_state=0
                                      )

plt.matshow(A=data,
            cmap=plt.cm.Blues
            )

plt.title(label="Original dataset")

# shuffle clusters
rng = np.random.RandomState(0)
row_idx = rng.permutation(data.shape[0])
col_idx = rng.permutation(data.shape[1])
data = data[row_idx][:, col_idx]

plt.matshow(A=data,
            cmap=plt.cm.Blues
            )

plt.title(label="Shuffled dataset")

model = SpectralCoclustering(n_clusters=5,
                             random_state=0
                             )

model.fit(data)

score = consensus_score(a=model.biclusters_,
                        b=(rows[:, row_idx], columns[:, col_idx])
                        )

print("consensus score: {:.3f}".format(score))

fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(A=fit_data,
            cmap=plt.cm.Blues
            )

plt.title("After biclustering; rearranged to show biclusters")

plt.show()
