import numpy as np
import matplotlib.pyplot as plt

imageFolder = 'dataset'
featuresFolder = 'features'

annotations = np.load('%s/annotations_train2017.npy' % (featuresFolder), allow_pickle=True).item()
annotations.update(np.load('%s/annotations_val2017.npy' % (featuresFolder), allow_pickle=True).item())


def showImage(imageId):
    imageName = '000000%s.jpg' % (imageId)
    try:
        image = plt.imread('%s/%s/%s' % (imageFolder, 'train2017', imageName))
    except:
        image = plt.imread('%s/%s/%s' % (imageFolder, 'val2017', imageName))
    plt.imshow(image)
    plt.show()


def showAnnotations(imageId):
    print("\nAnnotation for image %s:" % imageId)
    for annotation in annotations[imageId]:
        print(annotation)


def show(imageIds, image=True, annotation=True):
    if type(imageIds) is not list:
        imageIds = [imageIds]

    for imageId in imageIds:
        if annotation:
            showAnnotations(imageId)
        if image:
            showImage(imageId)

if __name__ == '__main__':
    show(['329084'])
