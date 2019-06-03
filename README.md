# CCA used for Image to Tag and Tag to Image Search

Implementation of this paper:
http://slazebni.cs.illinois.edu/publications/yunchao_cca13.pdf
Using CNN features.

---

You will need to have the MS COCO database and give path to the db folder in some python files.

**Make sure to use python 3**

### 1. Install dependencies
Install pycocotools and other python dependencies
```sh
git clone https://github.com/pdollar/coco.git && cd coco/PythonAPI && make install && pip3 install -e .
pip3 install --upgrade gensim nltk numpy torch torchvision tqdm Pillow scikit-image
python3 -m nltk.downloader stopwords
```

### 2. Extract image and text features from COCO
```sh
mkdir features
python3 Helper/extractTextFeatures.py --dataset-path /path/to/coco
python3 Helper/extractImageFeatures.py --dataset-path /path/to/coco
```

### 3. Compute CCA
```sh
python3 main.py --output CCA_0
```

### 4. Test CCA
Try:
```sh
python3 I2T.py --name CCA_0.npy
# or
python3 T2I.py --name CCA_0.npy
```
