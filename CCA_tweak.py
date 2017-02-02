# -*- coding: utf-8 -*-
from CCA_utils import *
from CCA_search import textToImageSearch, imageToTagSearch
from loadFeatures import *
from showImages import show
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


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

## Tag (text) features
T = get_computed_tag_features('Features/tagFeatures_COCO.npy')
d2 = T.shape[1]
phi_T = mapTagFeatures(T)


### COMPUTE COVARIANCE AND SOLVE CCA
print("Computing Covariance Matrix")

S, S_D = computeCovMatrix([phi_V, phi_T])


def evalCCA(dimensionCCA, eigPower, regularization):

    print("\nComputing CCA for (%d, %0.2f, %0.2f)" % (dimensionCCA, eigPower, regularization))
    W, D = solveCCA(S, S_D, dimensionCCA, regularization, eigPower)
    W_V = W[:d1]
    W_T = W[d1:]
    ap = []

    print("Computing AP For Text To Image Search")
    searchCategoryTerms = get_categories()

    for search in searchCategoryTerms:
        resIds, similarities = textToImageSearch(search, W_T, D, 0, phi_V, W_V, imIds)
        correctsIds = findImagesWithWordInCategory(search)
        truth = np.in1d(resIds, correctsIds)
        precision, recall, thresholds = precision_recall_curve(truth, similarities)
        average_precision = average_precision_score(truth, similarities)

        ap.extend([average_precision])

    aps.extend([1. * sum(ap) / len(ap)])
    return 1. * sum(ap) / len(ap)


aps = []
print "With 0 : %f" % evalCCA(60, -1, 0)
print "With 1 : %f" % evalCCA(60, -1, 1)
print "With 2 : %f" % evalCCA(60, -1, 2)
print "With 5 : %f" % evalCCA(60, -1, 5)
plt.plot(aps)
plt.show()
