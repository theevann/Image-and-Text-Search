from pycocotools.coco import COCO
from nltk.corpus import stopwords as sw
import numpy as np

dataDir = '/media/evann/Data/MS COCO/'
# dataType = 'train2014'
dataType = 'val2014'
annFile = '%s/annotations/captions_%s.json' % (dataDir, dataType)

coco_caps = COCO(annFile)
annIds = coco_caps.getAnnIds()
anns = coco_caps.loadAnns(annIds)
stopwords = sw.words("english")

print("Grouping annotations...")
annotations = {}
for ann in anns:
    imId = str(ann['image_id']).zfill(6)
    if imId in annotations:
        annotations[imId] = np.append(annotations[imId], ann["caption"])
    else:
        annotations[imId] = np.array(ann["caption"])

print("Extracting words from annotations...")

def sentenceToWords(sentence):
    words = sentence.lower().replace(',', '').replace('.', '').replace('"', '').replace("'s", '').split()
    words = [word for word in words if word not in stopwords]
    return words

wordAnnotations = {imId: sentenceToWords(' '.join(sentences)) for imId, sentences in annotations.items()}

print("Saving annotations...")
np.save('annotations_%s' % dataType, annotations)
np.save('wordsOfAnnotations_%s' % dataType, wordAnnotations)
