import os
import glob
import re
import numpy as np
import sys

LOG_PATHS = os.path.join('log', 'analysis_test_accuracy', '*.log')

files = glob.glob(LOG_PATHS)

best_acc_list = []
for file in files:
    best_acc = 0
    f = open(file, 'r')
    for line in f:
        matchObj = re.match(r'.*test_ce_loss:\s*(\d*\.?\d*),\s*acc:\s*(\d*\.?\d*).*', line)
        if matchObj is not None:
            matched_acc = float(matchObj.group(2))
            if best_acc < matched_acc:
                best_acc = matched_acc
    best_acc_list.append(best_acc)
    f.close()

print('best accuracy list: {}'.format(str(best_acc_list)))
print('accuracy mean: {}'.format(np.mean(best_acc_list)))
print('accuracy standard deviation: {}'.format(np.std(best_acc_list)))
