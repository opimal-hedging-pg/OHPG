from source.utils import Data_heston_test
from source.utils import Heston_price
import time
import numpy as np
import matplotlib.pyplot as plt
import json

from argparse import ArgumentParser

N = 2000

time_start = time.time()



for n in range(N):

    if n % 10 == 0:
        print(f'Ep: {n}')

    s = str(np.random.uniform()).split('.')[-1][:6]

    fn = f'data/heston1/heston-{s}'
    data, meta = Data_heston_test(N = 1)

    np.save(fn, np.array(data))


time_end = time.time()

# Save meta data

with open(f'data/heston1/meta.json', 'w') as f:
    json.dump(meta, f, indent = 4)
    f.close()

print('time cost: ', time_end - time_start, 's')
