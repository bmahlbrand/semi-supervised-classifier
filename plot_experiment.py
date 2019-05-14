import argparse
import json

from utils import viz_utils as viz

parser = argparse.ArgumentParser(description='PyTorch script for plotting results of experiments')
parser.add_argument('--experiment', type=str, default='', help='experiment json file to reload experiments')
parser.add_argument('--output', type=str, default='', help='output file location and prefix for plots')

args = parser.parse_args()

config = args.__dict__

# config['experiment'] = "experiments/experiment_2019-04-29[16_38_38].json"

output_file = config['output']

with open(config['experiment'], 'r') as f:
    experiment_data = json.load(f)

history = experiment_data['history']
# training_loss = history['training_loss']
# # validation_loss = history['validation_loss']
# validation_accuracy_top_1 = history['validation_accuracy_top_1']
# validation_accuracy_top_k = history['validation_accuracy_top_k']

network = experiment_data['network']
optimizer = experiment_data['optimizer']
learning_rate = experiment_data['learning_rate']
augment = experiment_data['augment']
session = experiment_data['logpath']
subset_sampler_size = experiment_data['subset_sampler_size']

session = session.replace("log/log_", "")

data_to_plot = []

# data_to_plot.append([{"name": "Training Loss", "values": training_loss}])
# data_to_plot.append([{"name": "Validation Loss", "values": validation_loss}])
# data_to_plot.append([{"name": "Validation Loss", "values": validation_loss},
#                      {"name": "Training Loss", "values": training_loss}])
#
# for item in data_to_plot:
#     viz.plot_png(item, output_file, network=network, optimizer=optimizer, lr=learning_rate,
#                  sampler_size=subset_sampler_size, augment=augment,
#                  session=session)


viz.plot_png([{"name": "Reconstruction Loss", "values": history['reconstruction_loss']}], output_file, network=network,
             optimizer=optimizer, lr=learning_rate, sampler_size=subset_sampler_size, augment=augment, session=session)

# plt.plot(range(len(history['losses'])), history['losses'], 'g-')
# plt.xlabel('batch steps')
# plt.ylabel('loss')
# plt.savefig('test_loss.png')

# plt.plot(range(args.epochs), history['validation_accuracy'], 'r-')
# plt.xlabel('epoch')
# plt.ylabel('validation accuracy')
# plt.savefig('valid_accuracy.png')
