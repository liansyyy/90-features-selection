import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from FS.de import jfs  # change this to switch algorithm
import matplotlib.pyplot as plt

# load data
data = pd.read_csv(r'../data/CUS-90-5000-label.csv').values
feat = np.asarray(data[:, 0:-1])
label = np.asarray(data[:, -1] - 1)

# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label, random_state=1)
fold = {'xt': xtrain, 'yt': ytrain, 'xv': xtest, 'yv': ytest}

# parameter
k = 1  # k-value in KNN
N = 20  # number of chromosomes
T = 100  # maximum number of generations
CR = 0.8
MR = 0.1
opts = {'k': k, 'fold': fold, 'N': N, 'T': T, 'CR': CR, 'MR': MR}

# perform feature selection
fmdl = jfs(feat, label, opts)
sf = fmdl['sf']

from clf.model import classifier

classifier(sf)

# number of selected features
num_feat = fmdl['nf']
print("Feature Size:", num_feat)
print("Feature : ", fmdl['sf'])

# plot convergence
curve = fmdl['c']
curve = curve.reshape(np.size(curve, 1))
x = np.arange(0, opts['T'], 1.0) + 1.0

fig, ax = plt.subplots()
ax.plot(x, curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Fitness')
ax.set_title('DE-90-1')
ax.grid()
plt.show()
