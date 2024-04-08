import tensorflow as tf
import argparse
import os
import numpy as np

#Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--state_dataset", default="input/state_txt", help="the state inputs")
parser.add_argument("--nstate_dataset", default="input/nstate_txt", help="the next state inputs")
parser.add_argument("--action_dataset", default="input/action_txt", help="the action inputs")
parser.add_argument("--mode", default="train", choices=["train", "test"], required=True)
parser.add_argument("--trained_model_dir", help="where to save/restore the model")

parser.add_argument("--max_episodes", type=int, default=1000, help="the number of training episodes")

parser.add_argument("--print_freq", type=int, default=50, help="print current reward and loss every print_freq iterations, 0 to disable")

args = parser.parse_args()

def get_shuffle_idx(num, batch_size):
  tmp = np.arange(num)
  np.random.shuffle(tmp)
  split_array = []
  cur = 0
  while num > batch_size:
    num -= batch_size
    if(num != 0):
      split_array.append(cur+batch_size)
      cur+=batch_size
  return np.split(tmp, split_array)
