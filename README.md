CS231N Gesture Recognition
Authors (in random order): Ying Hang Seah, Kenny Leung, Hiroshi Mendoza

We aim to create a gesture recognition algorithm using deep convolutional network and LSTM. 

## Getting Started

To get started, ensure you are running Python 3.6+ with all the required packages installed. Once all the requirements are installed, download the dataset and you can start training the model.

### Prerequisites

To create a virtualenv, run
```
pip3 install virtualenv
virtualenv .env
```

To activate the virtualenv, run
```
source .env/bin/activate
```

To deactivate the virtualenv, run
```
deactivate
```

Install the requirements while the virtualenv is activated by running
```
pip3 install -r requirements.txt
```

## Preparing the Dataset

To train our model, we are using [Chalearn Isolated Gesture](http://chalearnlap.cvc.uab.es/dataset/21/description/) dataset which contains 47933 RGB and RGB-D gesture videos with 249 different gestures. Download the dataset from Chalearn's website and the labels from [here](http://www.cbsr.ia.ac.cn/users/jwan/database/IsoGD_labels.zip). Extract the training, validation and test set and their respective labels in the following manner.
```
../
├── cs231n-sl-classification
│   └ ...
├ train.txt
├ train
│   ├ 001
│   │   ├ K_00001.avi
│   │   └ ...
│   └ ...
├ valid.txt
├ valid
│   ├ 001
│   │   ├ K_00001.avi
│   │   └ ...
│   └ ...
├ test.txt
├ test
│   ├ 001
│   │   ├ K_00001.avi
│   │   └ ...
│   └ ...
```
# TODO Eun Hiroshi's code to extract the videos into frames / combine it with prepare_dataset.py

Next, run prepare_dataset.py
```
python prepare_dataset.py
```
The dataset should be rearranged to the following structure
```
../
├── cs231n-sl-classification
│   └── ...
├ dataset
│   ├ train
│   │   ├ 001 (Label)
│   │   │   ├ K_00001.avi
│   │   │   ├ K_00001
│   │   │   │   ├ K_00001_1.png
│   │   │   │   └ ...
│   │   ├ 002
│   │   │   └ ...
│   │   └ ...
│   ├ valid
│   │   └ ... (Same structure as train)
│   ├ test
│   │   └ ... (Same structure as train)
```

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc