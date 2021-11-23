# CTRNet
The Pytorch implementation of CTRNet. (Conceptual text region network: Cognition-inspired accurate scene text detection)

Link: https://doi.org/10.1016/j.neucom.2021.08.026

Disclaimer: I proudly share with you this piece of code, so that you may reduce 90% of your work implementing CTRNet. The code is not production ready. Please understand that this project is only for research use, since we are not motivated to optimize it towards production purposes. As such, there may be some inconvinence when you implement CTRNet. For example, cache.py is needed even for testing. The caching process takes ~5 minutes in a multicore computing server, and may take a lot more time on a personal device. Also, the results of cache.py will take up a lot of space.
