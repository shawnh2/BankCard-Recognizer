# BankCard-Recognizer
Extracting bank-number from bankcard, based on Deep Learning with Keras.

Including auto and manual location, number identification, with GUI.

![bankcard](https://github.com/ShawnHXH/BankCard-Recognizer/blob/master/gui/icon/bankcard.png)


## Roadmap
* data augmentation
* cnn_blstm_ctc
* /manual locate
* gui

## Requirement
Python 3.6, Tensorflow-gpu, Keras, Cython, OpenCV2, Numpy, Scipy, PyQt5, clipboard.

## Environment
My platform is Win10 with Anaconda, NVIDIA GTX 1050.

## Usage
1. Open the whole project with PyCharm (Mine is 2018.3), then run demo.py.
2. Or `cd` to this project dir, then run `python ./demo.py`.
3. Will appear a GUI, press `Load` button to load one image about bankcard.
4. Press `Identify` button, it will start locate and do identification.
5. Activate manual location by double click the Image view, then draw the interest area and press `Identify`.
6. Also can adjust the image by pressing `Rotate` or `Zoom` button.

## Training
1. Unzip my dataset in ./dataset/dataset_for_crnn.py.
2. Run the `run.py` in crnn, and you can change some parameters depends on your hardware.
3. If you want use your own dataset, put it into `dataset/`.
4. Change the src_dir in `run.py`.

## About
### data augmentation
I wrote some functions about data augmentation, especially for data like image.

It contains shift, zoom, shear, rotate, resize, fill etc. Some are using `Scipy.ndimage`, some are built with `Cython`.
If you want to add you own cython code, write in ctoolkits.pyx and execute `python setup.py build_ext --inplace` in command line.

Here are some effects:

![data-aug-effect2](https://github.com/ShawnHXH/BankCard-Recognizer/blob/master/aug/effects/data-aug2.png)

### cnn_blstm_ctc
The model I used, refer to CNN_RNN_CTC. The CNN part is used VGG, with BLSTM as RNN and CTC loss.

This model's behave is pretty well. But training it takes time. In my case, `epochs=100, batch_size=16, aug_nbr=50, steps_per_epoch=200`
spends almost 4-5 hours. If you have a nice GPU, you will do better which I'm not :( .

The model's preview:

![model]()

### /manual locate

Auto locate is achieved by ...

Manual locate is achieved by inheriting from QGraphicView and QGraphicPixmapItem, overriding methods like mousePressEvent, 
mouseMoveEvent, mouseReleaseEvent and etc. This part only available in GUI.

![manual-locate]()

### gui
Using QtDesigner to design UI, and PyQt5 to finish other works.

## Reference

