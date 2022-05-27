from settings import LOGDIR, CHECKPOINTS_DIR
import os
from functools import reduce
import tensorflow as tf
import random, string


def get_latest_dir(directory):
    sorted_files = sorted(
        int(f.split('_')[0]) for f in os.listdir(directory) if 'json' not in f if '.DS_Store' not in f and 'overall' not in f)
    if not sorted_files:
        new_filename = '1'
    else:
        new_filename = str(int(sorted_files[-1]) + 1)
    return new_filename

def get_random_string():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

def get_new_model_log_paths():
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    if not os.path.exists(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)
    new_filename_logs = get_latest_dir(LOGDIR)
    new_filename_checkpoints = get_latest_dir(CHECKPOINTS_DIR)
    real_filename = str(max([int(new_filename_logs), int(new_filename_checkpoints)]))
    real_filename = real_filename + '_' + get_random_string()
    return os.path.join(LOGDIR, real_filename), os.path.join(CHECKPOINTS_DIR, real_filename)




def factors(n):
    return set(reduce(list.__add__,
                      ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
