#!/bin/sh

git clone https://github.com/theevann/Image-and-Text-Search.git && cd Image-and-Text-Search

mkdir dataset && cd dataset
wget http://images.cocodataset.org/zips/train2017.zip && unzip -q train2017.zip && rm train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip && rm val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -q annotations_trainval2017.zip && rm annotations_trainval2017.zip

cd ../.. && git clone https://github.com/pdollar/coco.git && cd coco/PythonAPI && sudo make install && pip install -e .
pip install --upgrade gensim nltk numpy torch torchvision tqdm Pillow scikit-image
python -m nltk.downloader stopwords
