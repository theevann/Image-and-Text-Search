from pycocotools.coco import COCO
import numpy as np

### GIVE DATA DIRECTORY
dataDir = '/media/evann/Data/MS COCO/'

### CHOOSE DATA TYPE
# dataType = 'val2014'
# dataType = 'train2014'
annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)

coco = COCO(annFile)
imgs = coco.loadImgs(coco.getImgIds())
cats = coco.loadCats(coco.getCatIds())

cat2im = {cat['name']: [str(imgId).zfill(6) for imgId in coco.getImgIds(catIds=cat['id'])] for cat in cats}
im2cat = {}

for cat in cats:
    imIds = coco.getImgIds(catIds=cat['id'])
    for imId in imIds:
        imId = str(imId).zfill(6)
        if imId in im2cat:
            im2cat[imId].extend([cat['name']])
        else:
            im2cat[imId] = [cat['name']]

np.save('cat2im_%s' % dataType, cat2im)
np.save('im2cat_%s' % dataType, im2cat)
