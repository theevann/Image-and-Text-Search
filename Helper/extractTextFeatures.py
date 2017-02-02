from gensim.models import Word2Vec
from pycocotools.coco import COCO
from nltk.corpus import stopwords
import numpy as np

dataDir = '/media/evann/Data/MS COCO/'
dataType = 'train2014'
annFile = '%s/annotations/captions_%s.json' % (dataDir, dataType)

coco_caps = COCO(annFile)
annIds = coco_caps.getAnnIds()
anns = coco_caps.loadAnns(annIds)

print("Creating sentences...")
cachedStopWords = stopwords.words("english")
sentences = [ann["caption"].lower().replace(',', '').replace('.', '').replace('"', '').split() for ann in anns]
sentences = [[word for word in sentence if word not in cachedStopWords] for sentence in sentences]

print("Learning model...")
model = Word2Vec(sentences, size=100, min_count=5, workers=4)
model.init_sims(replace=True)

print("Saving model and frequencies...")
wordsVectors = {word: model[word] for word in model.vocab.keys()}
np.save('wordFeatures', wordsVectors)

totalCount = sum(len(x) for x in sentences) * 1.
wordsFreqs = {word: model.vocab[word].count / totalCount for word in model.vocab.keys()}
np.save('wordFrequencies', wordsFreqs)

model.save('w2v_model')
