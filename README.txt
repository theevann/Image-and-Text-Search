CCA used for Image to Tag and Tag to Image Search

Implementation of this paper:
http://slazebni.cs.illinois.edu/publications/yunchao_cca13.pdf
Using CNN features...

===============
For linux users:

Download Pycocotools (not necessary for testing)
git clone https://github.com/pdollar/coco.git
Run 'make install' in the directory PythonApi


Install gensim (not necessary for testing)
pip install --upgrade gensim


Install Caffe by following these tutorials: (necessary for testing Image To Tag)
http://caffe.berkeleyvision.org/install_apt.html
http://installing-caffe-the-right-way.wikidot.com/start
Don't forget to 'make pycaffe' and to have a caffe folder in the repository
And then download googlenet neural network by using this command in caffe directory:
scripts/download_model_binary.py ./models/bvlc_googlenet


To try it, you will need to extract manually annotations and images Feature vectors using functions provided in the helper folder (to big to upload). You will also need to have the MS COCO database and give path to the db folder in some .py files.

I2T.py and T2I.py are respectively functions to test image to tag and tag to image search.
