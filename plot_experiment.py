import argparse
import json

from utils import viz_utils

parser = argparse.ArgumentParser(description='PyTorch script for plotting results of experiments')
parser.add_argument('--config', type=str, default='', help='config json file to reload experiments')
parser.add_argument('--output', type=str, default='output.png', help='output location for plots')

args = parser.parse_args()

with open(args.config, 'r') as f:
    experiment_data = json.load(f)

history = experiment_data['history']

# plt.plot(range(len(history['losses'])), history['losses'], 'g-')
# plt.xlabel('batch steps')
# plt.ylabel('loss')
# plt.savefig('test_loss.png')

# plt.plot(range(args.epochs), history['validation_accuracy'], 'r-')
# plt.xlabel('epoch')
# plt.ylabel('validation accuracy')
# plt.savefig('valid_accuracy.png')