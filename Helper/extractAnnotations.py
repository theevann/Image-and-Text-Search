from pycocotools.coco import COCO
import numpy as np

dataDir = '/media/evann/Data/MS COCO/'
# dataType = 'train2014'
# dataType = 'val2014'
annFile = '%s/annotations/captions_%s.json' % (dataDir, dataType)

coco_caps = COCO(annFile)
annIds = coco_caps.getAnnIds()
anns = coco_caps.loadAnns(annIds)

print("Grouping annotations...")
annotations = {}
for ann in anns:
    imId = str(ann['image_id']).zfill(6)
    if imId in annotations:
        annotations[imId] = np.append(annotations[imId], ann["caption"])
    else:
        annotations[imId] = np.array(ann["caption"])

print("Saving annotations...")
np.save('imgAnnotations_%s' % dataType, annotations)
