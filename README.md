# CTRNet
The Pytorch implementation of CTRNet. (Conceptual text region network: Cognition-inspired accurate scene text detection)<br />

**Link**: https://doi.org/10.1016/j.neucom.2021.08.026<br />

**Disclaimer**: I proudly share with you this piece of code, so that you may reduce 90% of your work implementing CTRNet. The code is not production ready. Please understand that this project is only for research use, since we are not motivated to optimize it towards production purposes. As such, there may be some inconvinence when you implement CTRNet. For example, cache.py is needed even for testing. The caching process takes ~5 minutes in a multicore computing server, and may take a lot more time on a personal device. Also, the results of cache.py will take up a lot of space.

## Requirements:
**System:** Ubuntu 20.04.2 LTS<br />
**Dataset:** CTW1500 https://1drv.ms/u/s!Aplwt7jiPGKilH4XzZPoKrO7Aulk<br />
**Dependency:**<br />
```
conda create -n ctrnet python == 3.6<br />
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch  # pytorch >= 1.2.0 is required<br />
pip install bentley_ottmann==0.1.0<br />
pip install dendroid==0.4.0  # Apparently bentley_ottmannn==0.1.0 requires dendroid==0.4.0<br />
pip install triangle<br />
pip install scikit-fem<br />
pip install matplotlib<br />
pip install opencv-python<br />
pip install pyclipper<br />
pip install sklearn<br />
pip install scipy<br />
pip install pillow<br />
pip install numpy<br />
```

## Caching
```
python cache.py -name ctw1500_train<br />
```

## Data Preprocessing
```
python preprocess.py -name ctw1500_train<br />
python preprocess.py -name ctw1500_test<br />
```

## Training and Testing
```
python train.py<br />
python test.py<br />
```
