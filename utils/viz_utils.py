import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve():
    pass

# plot_loss_change(learn.sched, sma=20)


def plot_png(data, file_name, network, optimizer, lr, sampler_size, augment, session, ylabel="loss"):

    data_names = []

    for item in data:
        data_name = item['name']
        data_names.append(data_name)
        data_values = item['values']
        plt.plot(range(1, len(data_values) + 1), data_values, label=data_name)

    title = f"{network.upper()} using {optimizer.upper()} with a learning rate of {lr}, a sample size of {sampler_size}"
    if augment:
        title += f", and augmentation"

    report_title = " / ".join(data_names)

    plt.suptitle(report_title)
    plt.title(title, fontsize=10)
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.xticks(np.arange(1, len(data_values) + 1, step=1))
    plt.legend()

    data_names.insert(0, session)
    if file_name:
        data_names.insert(1, file_name)

    output_file_name = "_".join(data_names).lower().replace(" ", "_")

    print(output_file_name)
    plt.savefig(output_file_name)
    plt.clf()


# https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
def plot_smooth_training_loss_change(train_losses, learning_rates, sma=1, n_skip=20, y_lim=(-0.01,0.01)):
    """
    Plots rate of change of the loss function.
    Parameters:
        train_losses - losses
        learning_rates - learning rate wrt to loss
        sma - number of batches for simple moving average to smooth out the curve.
        n_skip - number of batches to skip on the left.
        y_lim - limits for the y axis.
    """
    derivatives = [0] * (sma + 1)
    for i in range(1 + sma, len(learning_rates)):
        derivative = (train_losses[i] - train_losses[i - sma]) / sma
        derivatives.append(derivative)
        
    plt.ylabel("d/loss")
    plt.xlabel("learning rate (log scale)")
    plt.plot(learning_rates[n_skip:], derivatives[n_skip:])
    plt.xscale('log')
    plt.ylim(y_lim)


def plot_heatmap(heatmap):
    plt.imshow(heatmap)
    plt.show()

def scatterplot(data, name):
    plt.scatter(data[:,0], data[:,1])
    plt.title(name)
    # plt.legend(loc='best')
    plt.savefig(name)
    plt.clf()
    # plt.show()
