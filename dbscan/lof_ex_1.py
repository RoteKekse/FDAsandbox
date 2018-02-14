print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets.samples_generator import make_blobs

np.random.seed(10)

# Generate train data

centers = [[200, 5500]]
X_6, labels_true_1 = make_blobs(n_samples=20, centers=centers, cluster_std=50, # cluster_srd=0.25,0.1
                            random_state=0)
centers = [[2000, 6000]]
X_7, labels_true_1 = make_blobs(n_samples=20, centers=centers, cluster_std=50, # cluster_srd=0.25,0.1
                            random_state=0)
X_0 = np.random.rand(150, 2) * np.array([2500,1000])
X_1 = np.random.rand(50, 2) * np.array([1000,1500])
X_2 = np.random.rand(50, 2) * np.array([2500,7000])
X_3 = np.random.rand(50, 2) * np.array([1000,7000])
X_4 = np.random.rand(50, 2) * np.array([1500,3000])
X_5 = np.random.rand(75, 2) * np.array([1250,4000])
X = np.r_[X_0,X_1,X_2,X_3,X_4,X_5,X_6,X_7]

a = plt.scatter(X[:, 0], X[:, 1], c='white',
                edgecolor='k', s=20)
plt.axis('tight')
#plt.plot(X[:,0],X[:,1],'ro')
plt.xlim((0, 2500))
plt.ylim((0, 7000))
plt.grid(True)
plt.xlabel("Stunden seit letzter Transaktion")
plt.ylabel("Transaktionsvolumen (€)")
plt.show()

# Generate some abnormal novel observations
# X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
# X = np.r_[X + 3, X - 3, X_outliers]
#
# fit the model
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(X)
y_pred_outliers = y_pred[200:]
#
# # plot the level sets of the decision function
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Kreditkartenbeutzung")
#plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

a = plt.scatter(X[:430, 0], X[:430, 1], c='white',
                edgecolor='k', s=20)
b = plt.scatter(X[430:, 0], X[430:, 1], c='red',
                edgecolor='k', s=20)
plt.axis('tight')
#plt.plot(X[:,0],X[:,1],'ro')
plt.xlim((0, 2500))
plt.ylim((0, 7000))
plt.grid(True)
plt.xlabel("Stunden seit letzter Transaktion")
plt.ylabel("Transaktionsvolumen (€)")

plt.legend([a, b],
           ["normal observations",
            "abnormal observations"],
           loc="upper left")
plt.show()
