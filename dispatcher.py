from time import time
import random
import numpy as np
import os
import multiprocessing
from multiprocessing import Pool
import argparse
import subprocess
from itertools import product

manager = multiprocessing.Manager()
GPUqueue = manager.Queue()
processes_per_gpu = 1
for i in range(4):
    for _ in range(processes_per_gpu):
        GPUqueue.put(i)

def launch_experiment(args):
    args = [str(arg) for arg in args]
    subprocess.run(args=['python3', 'ib_vgg_distill_train.py', '--batch-norm',
        '--data-set', 'cifar10',
        '--resume-vgg-pt', 'baseline/cifar10/checkpoint_299_nocrop.tar',
        '--ban-crop',
        '--opt', 'adam',
        '--cfg', 'D4',
        '--epochs', '300',
        '--lr', '1.4e-3',
        '--weight-decay', '5e-5',
        '--kl-fac', '1.4e-5',
        '--save-dir', 'ib_vgg_distill/D4',
        '--distill_ratio', args[0],
        '--T', args[1],
        '--rand_seed', args[2]])
    launch_experiment.GPUq.put(int(os.environ['CUDA_VISIBLE_DEVICES']))
    return args

def distribute_gpu(q):
    launch_experiment.GPUq = q
    num = q.get()
    print("process id = {0}: using gpu {1}".format(os.getpid(), num))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(num)

def main():
    n_process = 4 * processes_per_gpu
    rand_seed_list = range(5)
    T_list = [0.5, 1, 2, 4]
    distill_ratio_list = [0, 0.2, 0.5, 0.7, 1]
    args_list = list(product(distill_ratio_list, T_list, rand_seed_list))
    print("# total training samples={}".format(len(args_list)))
    pool = Pool(processes=n_process,initializer=distribute_gpu,
            initargs=(GPUqueue,), maxtasksperchild=1)
    pool.map(launch_experiment, args_list)

main()
