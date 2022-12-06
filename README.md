# CTRNet

The Pytorch implementation of CTRNet.<br />

**Research Paper**: Conceptual text region network: Cognition-inspired accurate scene text detection<br />
**Link**: https://doi.org/10.1016/j.neucom.2021.08.026<br />

**Disclaimer**: We proudly share with you this piece of code, so that you may reduce 90% of your work implementing CTRNet. The code is not production ready. Please understand that this project is only for research use, since we are not motivated to optimize it towards production purposes. As such, there may be some inconvinence when you implement CTRNet. For example, cache.py is needed even for testing. The caching process takes ~5 minutes in a multicore computing server, and may take a lot more time on a personal device. Also, the results of cache.py will take up a lot of space.

## Requirements:
**System:** Ubuntu 20.04.2 LTS<br />
**Dataset:** CTW1500 https://1drv.ms/u/s!Aplwt7jiPGKilH4XzZPoKrO7Aulk<br />
**Dependency:**
```
conda create -n ctrnet python == 3.6
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch  # pytorch >= 1.2.0 is required
pip install bentley_ottmann==0.1.0
pip install dendroid==0.4.0  # Apparently bentley_ottmannn==0.1.0 requires dendroid==0.4.0
pip install triangle
pip install scikit-fem
pip install matplotlib
pip install opencv-python
pip install pyclipper
pip install sklearn
pip install scipy
pip install pillow
pip install numpy
```

## Caching
```
python cache.py -name ctw1500_train
```

## Data Preprocessing
```
python preprocess.py -name ctw1500_train
python preprocess.py -name ctw1500_test
```

## Training and Testing
```
python train.py
python test.py
```
