# -*- coding: utf-8 -*-
from CCA_search import tagsToImageSearch, imagesToTagSearch
from loadFeatures import *
from showImages import show
from sklearn.metrics import precision_recall_curve, average_precision_score
from random import sample
import matplotlib.pyplot as plt


### LOAD FEATURES
print("Loading features")

load_features("COCO")
W_T, W_V, phi_T, phi_V, D = np.load('Computed_CCA/CCA_1.npy', encoding='latin1')

imIds = get_images_id()


def plotAP(searchTerms, cat=False):
    ## AP for Text To Image Search
    print("\nPlotting AP Curves For Text To Image Search")
    ap = []

    resIds, similarities = tagsToImageSearch(searchTerms, W_T, D, 0, phi_V, W_V, imIds)

    for i, search in enumerate(searchTerms):
        correctsIds = findImagesWithWordInCategory(search) if cat else findImagesWithWordInAnnotations(search)
        truth = np.in1d(resIds[i], correctsIds)
        precision, recall, thresholds = precision_recall_curve(truth, similarities[i])
        average_precision = average_precision_score(truth, similarities[i])

        print('Average Precision for {0}: {1:0.2f}'.format(search, average_precision))
        ap.extend([average_precision])

        if interactiveCheck:
        ### Plot Precision-Recall curve
            lw = 2
            plt.clf()
            plt.plot(recall, precision, lw=lw, color='navy', label='Precision-Recall curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall for {0}: AUC={1:0.2f}'.format(search, average_precision))
            plt.legend(loc="lower left")
            plt.show()

    print('Average Global Precision : {0:0.2f}'.format(1. * sum(ap) / len(ap)))


def precisionAtRankT2I(searchTerms, rank, cat=False):
    print("\nComputing 'precision at rank' for Tag To Image Search over {0} images".format(len(searchTerms)))

    sumCorrects = 0.
    resIds, similarities = tagsToImageSearch(searchTerms, W_T, D, rank, phi_V, W_V, imIds)
    rank = rank if rank != 0 else len(imIds)

    for i, search in enumerate(searchTerms):
        correctsIds = findImagesWithWordInCategory(search) if cat else findImagesWithWordInAnnotations(search)
        truth = np.in1d(resIds[i], correctsIds)
        sumCorrects += sum(truth)
        print("Precision at rank {0} for {1} : {2}".format(rank, search, 1. * sum(truth) / rank))
    print("Precision at rank {0} for {1} words : {2}".format(rank, len(searchTerms), sumCorrects / (len(searchTerms) * rank)))


### AP for Image To Tag Search

# Select random images from val2014
# Perform image to text search for each of them
# Take 1 ranked term and check precision, ie. check if it is in the description
def precisionAtRankI2T(numberOfTestImages, maxRankCheck, imgIds=None):
    print("\nComputing 'precision at rank' for Image To Tag Search over {0} images".format(numberOfTestImages))

    testFeaturesDir = 'Features'
    testImagesVectors = np.load(testFeaturesDir + '/imgFeatures.npy', encoding='latin1').item()
    testAnnotations = np.load(testFeaturesDir + '/annotations.npy').item()
    testImgIds = list(testImagesVectors.keys())
    if imgIds is None:
        sampleIds = sample(testImgIds, numberOfTestImages)
    else:
        sampleIds = imgIds

    vecs = [testImagesVectors[sampleId][0] for sampleId in sampleIds]
    tags = imagesToTagSearch(vecs, W_V, D, maxRankCheck, phi_T, W_T, imIds)

    sumCorrects = np.zeros(maxRankCheck)
    for i, sampleId in enumerate(sampleIds):
        tagsImg = tags[i]
        correctsTags = np.unique(annotationsToWords([sampleId]))
        truthTags = np.in1d(tagsImg, correctsTags)
        sumCorrects += truthTags

    print('\n')
    for i in range(maxRankCheck):
        print("Precision at rank {0} : {1}".format(i+1, sumCorrects[i] / numberOfTestImages))


### TESTS

interactiveCheck = False

searchTerms = ['bus', 'cat', 'boat', 'street', 'bed', 'pizza', 'baseball', 'wave']


# Without using MS COCO categories

plotAP(searchTerms)
precisionAtRankT2I(searchTerms, 40)
precisionAtRankI2T(10, 5)


# Using MS COCO categories

searchCategoryTerms = get_categories()
plotAP(searchCategoryTerms, cat=True)
precisionAtRankT2I(searchCategoryTerms, 50, cat=True)
