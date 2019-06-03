import torch
from torchvision import models, transforms
from pycocotools.coco import COCO

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

batch_size = 250
num_workers = 8

annotationsFolder = "../dataset/annotations"
imageFoler = "../dataset"
featuresFolder = "../features"
dataFolders = ['train2017', 'val2017']

#############################################
# Create a custom Dataset to be able to use the Dataloader class and have multiple workers

class Transformer():
    def __init__(self):
        # Define image transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        expand = transforms.Lambda(lambda x: x.expand(3, 224, 224))
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize, expand])

    def __call__(self, image):
        return self.transform(image)


class CocoDataset(Dataset):
    def __init__(self, annotationsFolder, imageFoler, dataFolder):
        annotationsFile = '%s/instances_%s.json' % (annotationsFolder, dataFolder)
        coco = COCO(annotationsFile)
        self.folder = '%s/%s' % (imageFoler, dataFolder)
        self.imgs = coco.loadImgs(coco.getImgIds())
        self.transformer = Transformer()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        info = self.imgs[idx]
        img = self.transformer(Image.open("%s/%s" % (self.folder, info['file_name'])))
        return img, str(info['id']).zfill(6)

#############################################


def main():
    # Load Resnet-50 and remove last layer
    print("Loading Resnet-50")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet50 = models.resnet50(pretrained=True).to(device)
    model = torch.nn.Sequential(*(list(resnet50.children())[:-1]))

    # if multiple GPU availables
    model = torch.nn.DataParallel(torch.nn.Sequential(*(list(resnet50.children())[:-1])))

    torch.set_grad_enabled(False)

    for dataFolder in dataFolders:
        dataset = CocoDataset(annotationsFolder, imageFoler, dataFolder)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        print("\nRunning ResNet-50 over %s folder" % dataFolder)
        imgVectors = {}
        for batch, ids in tqdm(loader):
            out = model(batch.to(device)).squeeze().cpu()
            imgVectors.update({imgId: vec for imgId, vec in zip(ids, out)})

        print("Saving %s images features..." % dataFolder)
        torch.save(imgVectors, '%s/imgFeatures_%s' % (featuresFolder, dataFolder))


if __name__ == '__main__':
    main()
