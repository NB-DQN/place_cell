## Requirements

* NumPy (>= 1.6.1)
* SciPy (>= 0.9)
* Chainer
* (pyCUDA)

## How to train LSTM
```
python train_practice.py # CPU mode
python train_practice.py -g 0 # GPU mode (not that fast)
```

## Comments
* Throuput is approximately 40-80 epochs/sec on my PC (Core i5 2.9 GHz, 8GB memory)
* Best accuracy for test dataset is 70/100 (8/25)
