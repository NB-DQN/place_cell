import os
os.chdir('path/to/repo')

import train
model = train.pretrain() # 9*9 maze
model = train.pretrain(5,5) # 5*5 maze