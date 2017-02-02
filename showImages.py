import numpy as np
import matplotlib.pyplot as plt

imgsDir = '/media/evann/Data/MS COCO'
featuresDir = 'Features'
dataType = 'train2014'
annotations = np.load('%s/annotations.npy' % (featuresDir)).item()


def showImage(imId):
    imgName = 'COCO_%s_000000%s.jpg' % (dataType, imId)
    I = plt.imread('%s/%s/%s' % (imgsDir, dataType, imgName))
    plt.imshow(I)
    plt.show()


def showAnnotations(imId):
    print("\nAnnotation for image %s:" % imId)
    for ann in annotations[imId]:
        print(ann)


def show(imIds, img=True, ann=True):
    if type(imIds) is not list:
        imIds = [imIds]
    for imId in imIds:
        if ann:
            showAnnotations(imId)
        if img:
            showImage(imId)

if __name__ == '__main__':
    show(['329084'])
