## Requirements

* NumPy (>= 1.6.1),
* SciPy (>= 0.9),
* six,
* Chainer,
* os,

## How to train LSTM
```
import os
os.chdir('path/to/repo')

import train
model = train.pretrain() # 9*9 maze
model = train.pretrain(5,5) # 5*5 maze
```