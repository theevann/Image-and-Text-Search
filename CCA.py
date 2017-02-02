# -*- coding: utf-8 -*-
from CCA_utils import solveCCA, mapVisualFeatures, mapTagFeatures
from CCA_search import textToImageSearch, imageToTagSearch
from loadFeatures import *
from showImages import show


### LOAD FEATURES
print("Loading features")
# dbWordFeature = "WIKI"
dbWordFeature = "COCO"

load_features(dbWordFeature)
imIds = get_images_id()


## Visual features
V = get_visual_features()
d1 = V.shape[1]
phi_V = mapVisualFeatures(V)

## Text features
T = get_tag_features(fileName='Features/tagFeatures_COCO.npy')
d2 = T.shape[1]
phi_T = mapTagFeatures(T)

## We can save it...
# np.save('Features/tagFeatures_DATASET.npy', T)


### COMPUTE COVARIANCE AND SOLVE CCA
print("Computing CCA")

W, D = solveCCA([phi_V, phi_T], dimension=80, regularization=1, power=-1)

# Extract matrices
W_V = W[:d1]
W_T = W[d1:]

## We can save it...
# np.save('CCA_0', [W_T, W_V, phi_T, phi_V, D])


### TESTS ...
print("Testing")

search = 'yellow'
resIds, similarities = textToImageSearch(search, W_T, D, 10, phi_V, W_V, imIds)
tags, counts = imageToTagSearch(V[4001], W_V, D, 15, phi_T, W_T, imIds)


print("\n=============")
print("Tag To Images")
print("Input: %s" % search)
show(resIds.tolist())

print("\n=============")
print("Image To Tags")
print("Output: %s" % tags)
show(imIds[4001])
