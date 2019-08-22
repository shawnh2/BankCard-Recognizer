# BankCard-Recognizer

![mit](https://img.shields.io/github/license/mashape/apistatus.svg)

Extracting bank-number from bankcard, based on Deep Learning with Keras.

Including auto and manual location, number identification, with GUI.

![bankcard](https://github.com/ShawnHXH/BankCard-Recognizer/blob/master/gui/icon/bankcard.png)


## Roadmap
* data augmentation
* cnn_blstm_ctc
* EAST/manual locate
* gui

## Requirement
Python 3.6, Tensorflow-gpu, Keras, Cython, OpenCV2, Numpy, Scipy, PyQt5, PIL, clipboard.

## Environment
My platform is Win10 with Anaconda, PyCharm 2018.3, NVIDIA GTX 1050.

## Usage
1. Download trained model, [CRNN](https://pan.baidu.com/s/1Cyj1YHhHxlX-3Lgj0vQ35A) extracting-code:`6eqw`, 
[EAST](https://pan.baidu.com/s/1R-kD0HGTomS8O0JhXJ-hCA) extracting-code:`qiw5`. 
2. Then put CRNN model into `crnn/model`, put EAST model into `east/model`.
3. Run `python demo.py`.
4. In GUI, press `Load` button to load one image about bankcard or load from `dataset/test/`. 
5. Press `Identify` button, it will start locate and do identification.
6. Activate manual location by double click the Image view, then draw the interest area and press `Identify`.

## Training
### Prepare
Download my dataset, [CRNN](https://pan.baidu.com/s/1Ji0ZOv-rMSPcN2W6uO0K5Q) extracting-code:`1jax`,
[EAST](https://pan.baidu.com/s/1UL1OdLEL-uNRQl8d11NkeQ) extracting-code:`pqba`. and unzip dataset in `./dataset`.

### for CRNN
1. Run the `run.py` in crnn, and you can change some parameters depends on your hardware.
2. If you want use your own dataset, put it into `dataset/` and change the src_dir in `run.py`.
3. If doing data augmentation, it will take some time to generate `.npz` file, also recommend `aug_nbr` to be 30-80.
### for EAST
1. My dataset is collecting from Internet: Baidu, Google, and thanks [Kesci](https://www.kesci.com/home/dataset/5954cf1372ead054a5e25870). It has been labeld with ICDAR 2015 format, you can see it in `txt/`. This tiny dataset is unable to cover all the situation, if you have rich one, it may perform better.
2. If you would like to get more data, make sure data has been labeled, or you can take `dataset/tagger.py` to label it.
3. Modify `cfg.py`, see default values.
4. Run `python preprocess.py`. If process goes well, you'll see generated data like this:

![act_gt](https://github.com/ShawnHXH/BankCard-Recognizer/blob/master/readme/act_gt_img_99.png)

5. Finally, `python run.py`.

## About
### data augmentation
I wrote some functions about data augmentation, especially for data like image.

It contains shift, zoom, shear, rotate, resize, fill etc. Some are using `Scipy.ndimage`, some are built with `Cython`.
If you want to add you own cython code, write in ctoolkits.pyx and execute `python setup.py build_ext --inplace` in command line.

Here are some effects:

![data-aug-effect2](https://github.com/ShawnHXH/BankCard-Recognizer/blob/master/readme/data-aug2.png)

### cnn_blstm_ctc
The model I used, refer to CNN_RNN_CTC. The CNN part is using VGG, with BLSTM as RNN and CTC loss.

This model's behave is pretty well. But training it takes time. In my computer, `epochs=100, batch_size=16, aug_nbr=50, steps_per_epoch=200`
spends almost 4-5 hours. If you have a nice GPU, you will do better which I'm not :( .

The model's preview:

![model](https://github.com/ShawnHXH/BankCard-Recognizer/blob/master/readme/model.png)

### EAST/manual locate

Auto locate is using one famous Text Detection Algorithm - EAST. [See more details](https://zhuanlan.zhihu.com/p/37504120).

In this project, I prefer to use AdvancedEAST. It is an algorithm used for Scene-Image-Text-Detection, which is primarily based on EAST, and the significant improvement was also made, which make long text predictions more accurate. Original repo see Reference 1.

Also, training process is quiet quick and nice. As practical experience, img_size is better to be 384. The `epoch_nbr` is no longer important any more, for img_size like 384, usually training will early stop at epoch 20-22. But if you have a large dataset, try to play with these parameters.

This model's preview:

![model](https://github.com/ShawnHXH/BankCard-Recognizer/blob/master/readme/east.png)

Manual locate is only available in GUI. Here're some performance in .gif:

![manual-locate1](https://github.com/ShawnHXH/BankCard-Recognizer/blob/master/readme/manual-1.gif)

![manual-locate2](https://github.com/ShawnHXH/BankCard-Recognizer/blob/master/readme/manual-2.gif)

### gui
Using QtDesigner to design UI, and PyQt5 to finish other works.

## Statement
1. The bankcard images are collecting from Internet, if any images make you feel uncomfortable, please contact me.
2. If you have any issues, post it in Issues.

## Reference
1. https://github.com/huoyijie/AdvancedEAST
2. https://www.cnblogs.com/lillylin/p/9954981.html 
