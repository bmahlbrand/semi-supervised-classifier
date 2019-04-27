import argparse

from utils import viz_utils

parser = argparse.ArgumentParser(description='PyTorch script for plotting results of experiments')
parser.add_argument('--config', type=str, default='', help='config json file to reload experiments')
