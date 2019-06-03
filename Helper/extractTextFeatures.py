from gensim.models import Word2Vec
from pycocotools.coco import COCO
from nltk.corpus import stopwords as sw
import numpy as np
import torch
import re


annotationsFolder = "../dataset/annotations"
featuresFolder = "../features"

dataTypes = ['train2017', 'val2017']
stopwords = sw.words("english")


def sentenceToWords(sentence):
    words = re.sub('[\.,"]|\'s', '', sentence.lower()).split()
    return [word for word in words if word not in stopwords]


def wordsToVec(words, dict):
    return np.sum(dict[word] for word in words if word in dict)


def extractAnnotations(dict):
    for dataType in dataTypes:
        annFile = '%s/captions_%s.json' % (annotationsFolder, dataType)
        coco = COCO(annFile)
        anns = coco.loadAnns(coco.getAnnIds())

        print("Grouping annotations for %s ..." % dataType)
        annotations = {}
        for ann in anns:
            imageId = str(ann['image_id']).zfill(6)
            annotations[imageId] = np.append(annotations.get(imageId, np.array([])), ann["caption"])

        print("Extracting words from annotations for %s ..." % dataType)
        annotation2word = {imageId: sentenceToWords(' '.join(sentences)) for imageId, sentences in annotations.items()}
        annotationFeatures = {imageId: wordsToVec(words, dict) for imageId, words in annotation2word.items()}

        print("Saving annotations for %s ..." % dataType)
        np.save('%s/annotations_%s' % (featuresFolder, dataType), annotations)  # TODO Needed ?
        np.save('%s/annotation2word_%s' % (featuresFolder, dataType), annotation2word)
        np.save('%s/annotationFeatures_%s' % (featuresFolder, dataType), annotationFeatures)


def extractCategories():
    for dataType in dataTypes:
        annFile = '%s/instances_%s.json' % (annotationsFolder, dataType)
        coco = COCO(annFile)
        imgs = coco.loadImgs(coco.getImgIds())
        cats = coco.loadCats(coco.getCatIds())

        print("Extracting categories for %s ..." % dataType)

        cat2im = {}
        im2cat = {}

        for cat in cats:
            imageIds = [str(imageId).zfill(6) for imageId in coco.getImgIds(catIds=cat['id'])]
            cat2im[cat['name']] = imageIds
            for imageId in imageIds:
                im2cat.setdefault(imageId, []).append(cat['name'])

        np.save('%s/cat2im_%s' % (featuresFolder, dataType), cat2im)
        np.save('%s/im2cat_%s' % (featuresFolder, dataType), im2cat)


def learnW2VModel():
    dataType = dataTypes[0]
    annFile = '%s/captions_%s.json' % (annotationsFolder, dataType)
    coco = COCO(annFile)
    anns = coco.loadAnns(coco.getAnnIds())

    print("Creating sentences...")
    sentences = [sentenceToWords(ann["caption"]) for ann in anns]

    print("Learning Word2Vec model...")
    model = Word2Vec(sentences, size=100, min_count=5, workers=10)
    model.init_sims(replace=True)

    print("Saving model and frequencies...")
    wordsVectors = {word: torch.from_numpy(model[word]) for word in model.wv.vocab.keys()}
    np.save('%s/wordFeatures' % featuresFolder, wordsVectors)
    return wordsVectors


if __name__ == '__main__':
    wordsVectors = learnW2VModel()
    extractAnnotations(wordsVectors)
    extractCategories()
