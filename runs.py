import os
import numpy as np


alpha_s = np.linspace(0.1,1,5)
l1_ratio_s = np.linspace(0.1,1,5)

for a in alpha_s:
    for l1 in l1_ratio_s:
        print(f"logging experiment for p1:{a} and p2:{l1}")
        os.system(f"python demo.py -a {a} -l1 {l1}")