# Imports
# Python ≥3.5 is required
import sys

assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

# Common imports
import numpy as np
import pandas as pd
import os
import tarfile
import cv2
import pickle

# to make this notebook's output stable across runs
np.random.seed(42)


def target_value(val):
    if val == 'aanwezig':
        return 2
    if val == 'buiten':
        return 1
    return 0


if not os.path.isdir('./model/'):
    os.mkdir('./model/')

#Extract when not already extracted
if not os.path.isdir('./data/classificatie'):
    if not os.path.isfile('./data/classificatie.tar'):
        raise Exception('Classificatie.tar not fount')

    print('Extracting tar...')
    tar = tarfile.open('./data/classificatie.tar')
    tar.extractall('./data/')
    tar.close()
    print('Extracting tar Done!')

if not os.path.isdir('./data/classificatie'):
    raise Exception('Extracted files not found')

samples = []

# Get grayscale values from pictures
print('Creating dataframe')
for folder in os.listdir('./data/classificatie/'):
    for file in os.listdir(f'./data/classificatie/{folder}'):

        img = cv2.imread(f'./data/classificatie/{folder}/{file}', 0)
        img = cv2.resize(img, (320, 176))

        # add them to a dataframe
        imgd = dict()
        imgd['target'] = target_value(folder)
        c = 0
        for i in img.flatten():
            c += 1
            imgd[f'p{c}'] = i
        samples.append(imgd)
        print(file)

musti = pd.DataFrame.from_records(samples)

print('Dataframe Created')

X, y = musti.drop('target', axis=1), musti['target']
y = y.astype(np.uint8)  # less RAM space

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print('Start training...')
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train)
print('End training!')

model = sgd_clf # Generated machine learning model (fit result)
pickle.dump(model, open('./model/model.sav', 'wb'))

print(cross_val_score(model, X_test, y_test, cv=3))







