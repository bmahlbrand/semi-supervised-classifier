import matplotlib.pyplot as plt

def plot_learning_curve():
    pass

# plot_loss_change(learn.sched, sma=20)


def plot_training_loss(train_losses, name):
    name = name + "training-loss.png"
    plt.plot(range(len(train_losses)), train_losses, 'b-')
    plt.xlabel('train_step')
    plt.ylabel('training_loss')
    plt.savefig(name)
    plt.clf()


def plot_training_loss_validation_accuracy(train_losses, validation_accuracy, name):
    name = name + "-training-loss-validation-accuracy.png"
    plt.plot(range(len(train_losses)), train_losses, 'b-', label="training loss")
    plt.plot(range(len(validation_accuracy)), validation_accuracy, 'y-', label="validation accuracy")
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(name)
    plt.clf()


def plot_training_loss_validation_loss(train_losses, validation_loss, name):
    name = name + "-training-loss-validation-loss.png"
    plt.plot(range(len(train_losses)), train_losses, 'b-', label="training loss")
    plt.plot(range(len(validation_loss)), validation_loss, 'g-', label="validation loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(name)
    plt.clf()


def plot_validation_loss(validation_losses, name):
    name = name + "-validation-loss.png"
    plt.plot(range(len(validation_losses)), validation_losses, 'b-', label='validation losses')
    plt.xlabel('training_step')
    plt.ylabel('validation_loss')
    plt.legend()
    plt.savefig(name)
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


def plot_validation_learning_curve(test_err, name):
    name = name + "-validation-accuracy.png"
    plt.plot(range(len(test_err)), test_err, 'r-')
    plt.xlabel('epoch')
    plt.ylabel('test_error_rate')
    plt.savefig(name)
    plt.clf()
    # plt.show()

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
