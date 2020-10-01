import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Main settings
_bool_mass_convergence = False  # flag to allow for counting regularization
version_name = "noMassConvergence_"

# Set random seeds (tensorflow and numpy)
seed = 41
tf.compat.v1.set_random_seed(seed)
np.random.seed(seed)

# Simulation settings
n_points = 3  # number of objects
n_points_pred = 50  # number of predictions
smoothing_lambda = 0.2  # bandwidth

# Other settings
n_iter = 100000 + 1
alpha = 0.1

# Fixed values
visualization_step = 200
batch_size, n_channel = 1, 1
max_occurrence = n_points + 2


## Create directories to save images
if not os.path.exists("img"):
    os.makedirs("img")

if not os.path.exists("img"):
    os.makedirs("zoom")


## Define continuous heatmap matching loss function
def loss_func(location_pred, location_true):
    x_true, y_true = location_true[:, 0], location_true[:, 1]  # tf split
    x_pred, y_pred, p_pred = location_pred[:, 0], location_pred[:, 1], location_pred[:, 2]  # tf split
    len_true, len_pred = np.shape(x_true)[0], np.shape(x_pred)[0]

    term1 = np.pi * np.square(smoothing_lambda) / 2 * np.exp(
        -(np.square(
            np.tile(x_true[np.newaxis, :], [len_true, 1]) - np.tile(x_true[np.newaxis, :], [len_true, 1]).T) +
          np.square(
              np.tile(y_true[np.newaxis, :], [len_true, 1]) - np.tile(y_true[np.newaxis, :], [len_true, 1]).T)) /
        (2 * np.square(smoothing_lambda)))

    term2 = p_pred[np.newaxis, :] * p_pred[np.newaxis, :].T * np.pi * np.square(smoothing_lambda) / 2 * np.exp(
        -(np.square(
            np.tile(x_pred[np.newaxis, :], [len_pred, 1]) - np.tile(x_pred[np.newaxis, :], [len_pred, 1]).T) +
          np.square(
              np.tile(y_pred[np.newaxis, :], [len_pred, 1]) - np.tile(y_pred[np.newaxis, :], [len_pred, 1]).T)) /
        (2 * np.square(smoothing_lambda)))

    term3 = np.tile(p_pred[np.newaxis, :], [len_true, 1]) * np.pi * np.square(smoothing_lambda) / 2 * np.exp(
        -(np.square(
            np.tile(x_pred[np.newaxis, :], [len_true, 1]) - np.tile(x_true[np.newaxis, :], [len_pred, 1]).T) +
          np.square(
              np.tile(y_pred[np.newaxis, :], [len_true, 1]) - np.tile(y_true[np.newaxis, :], [len_pred, 1]).T)) /
        (2 * np.square(smoothing_lambda)))

    ll = np.sum(term1) + np.sum(term2) - 2 * np.sum(term3)

    return ll.astype(np.float32)


## Compute gradients of the loss
def loss_func_grad(op, grad):
    location_pred = op.inputs[0]
    location_true = op.inputs[1]

    x_true, y_true = location_true[:, 0], location_true[:, 1]  # tf split
    x_pred, y_pred, p_pred = location_pred[:, 0], location_pred[:, 1], location_pred[:, 2]  # tf split
    len_true, len_pred = np.shape(x_true)[0], np.shape(x_pred)[0]

    x_dist_pred_pred, y_dist_pred_pred = \
        tf.tile(x_pred[np.newaxis, :], [len_pred, 1]) - tf.transpose(tf.tile(x_pred[np.newaxis, :], [len_pred, 1])), \
        tf.tile(y_pred[np.newaxis, :], [len_pred, 1]) - tf.transpose(tf.tile(y_pred[np.newaxis, :], [len_pred, 1]))

    x_dist_pred_true, y_dist_pred_true = \
        tf.tile(x_pred[np.newaxis, :], [len_true, 1]) - tf.transpose(tf.tile(x_true[np.newaxis, :], [len_pred, 1])), \
        tf.tile(y_pred[np.newaxis, :], [len_true, 1]) - tf.transpose(tf.tile(y_true[np.newaxis, :], [len_pred, 1]))

    exp_dist_pred_pred = tf.math.exp(
        -(np.square(x_dist_pred_pred) + np.square(y_dist_pred_pred)) / (2 * np.square(smoothing_lambda)))

    exp_dist_pred_true = tf.math.exp(
        -(np.square(x_dist_pred_true) + np.square(y_dist_pred_true)) / (2 * np.square(smoothing_lambda)))

    xx = np.pi / 2 * p_pred * \
         (tf.math.reduce_sum(exp_dist_pred_true * x_dist_pred_true, axis=0)
          - tf.math.reduce_sum(tf.transpose(
                     tf.tile(p_pred[np.newaxis, :], [len_pred, 1])) * exp_dist_pred_pred * x_dist_pred_pred, axis=0))

    yy = np.pi / 2 * p_pred * \
         (tf.math.reduce_sum(exp_dist_pred_true * y_dist_pred_true, axis=0)
          - tf.math.reduce_sum(tf.transpose(
                     tf.tile(p_pred[np.newaxis, :], [len_pred, 1])) * exp_dist_pred_pred * y_dist_pred_pred, axis=0))

    pp = np.pi * np.square(smoothing_lambda) * \
         (tf.math.reduce_sum(
             tf.transpose(tf.tile(p_pred[np.newaxis, :], [len_pred, 1])) * exp_dist_pred_pred, axis=0)
          - tf.math.reduce_sum(exp_dist_pred_true, axis=0))

    gradients = tf.concat([tf.expand_dims(xx, axis=1), tf.expand_dims(yy, axis=1), tf.expand_dims(pp, axis=1)], axis=1)

    return gradients * grad, location_true * 0


# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1e8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


## Tensorflow Graph
y_data = tf.placeholder(dtype=tf.float32, shape=[n_points, 2])  # labels
predicted_points = tf.Variable(tf.random_normal([n_points_pred, 3]))  # predictions
predictions = tf.sigmoid(predicted_points)

loss_HM = py_func(loss_func, [predictions, y_data], [tf.float32], grad=loss_func_grad)[0]

if _bool_mass_convergence:

    ## Compute counting loss
    count_prediction = tf.one_hot(tf.zeros(batch_size * n_channel, dtype=tf.int32),
                                  max_occurrence)  # initial prediction

    mass = tf.slice(predictions, [0, 2], [n_points_pred, 1])
    mass = tf.unstack(mass)

    for output in mass:
        increment = tf.expand_dims(output, 0)

        count_prediction = tf.multiply(tf.concat((tf.tile(1 - increment, [1, max_occurrence - 1]),
                                                  tf.ones([batch_size * n_channel, 1])), axis=1), count_prediction) \
                           + tf.multiply(tf.tile(increment, [1, max_occurrence]), tf.slice(
            tf.concat((tf.zeros([batch_size * n_channel, 1]), count_prediction), axis=1), [0, 0],
            [batch_size * n_channel, max_occurrence]))

    loss_count = alpha * tf.reduce_mean(
        -tf.reduce_sum(tf.one_hot([n_points], max_occurrence) * tf.log(count_prediction + 1e-9),
                       reduction_indices=[1]))

    ## Loss function
    loss = loss_HM + loss_count
else:
    loss = loss_HM
    loss_count = tf.constant(0)

## Opmization through gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)

## Generate random ground-truth locations
points = np.random.random([n_points, 2]).astype(np.float32) * 0.9 + 0.05

## Loss history placeholder
ll_smooth_list = []
ll_count_list = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(n_iter):
        if step % 100:
            print("step:", step)

        ## Perform one step of the optimization
        _, pp, ll_smooth, ll_count = sess.run([train, predictions, loss_HM, loss_count], feed_dict={y_data: points})

        if step % 10 == 0:
            ## Save values of the losses
            ll_smooth_list.append(ll_smooth), ll_count_list.append(ll_count)

        if step < 100 or step % int(visualization_step / 10) == 0 or step < 100 or visualization_step == 0:
            ## Display the current situation
            f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(14, 8))
            a0.scatter(points[:, 0], points[:, 1], s=300, facecolors='none', edgecolors='k', linewidth=3, alpha=0.7)
            for kk in range(pp.shape[0]):
                a0.scatter(pp[kk, 0], pp[kk, 1], s=100, facecolors='b', alpha=pp[kk, 2])

            a0.text(0.95, 0.02, str(step), ha="right", fontweight="bold")
            a0.axes.get_xaxis().set_ticks([])
            a0.axes.get_yaxis().set_ticks([])

            a0.scatter(-0.1, -0.1, s=200, facecolors='none', edgecolors='k', linewidth=3, alpha=0.7,
                       label="Ground-Truth")
            a0.scatter(-0.1, -0.1, s=200, facecolors='b', alpha=0.7, label="Predictions")

            a0.legend(loc='upper left', fontsize=15)
            a0.set_xlim([0, 1])
            a0.set_ylim([0, 1])

            plt.setp(a0.spines.values(), color=3 * [0.4])
            plt.setp([a0.get_xticklines(), a0.get_yticklines()], color=3 * [0.4])

            a1.plot(100 * np.arange(len(ll_smooth_list[::10])), np.log(ll_smooth_list[::10]), 'r', label="Heatmap Loss")
            if _bool_mass_convergence:
                a1.plot(np.log(ll_count_list[::10]), 'b', label="Counting Loss")

            a1.legend(loc='upper right')
            a1.spines['right'].set_visible(False)
            a1.spines['top'].set_visible(False)

            plt.setp(a1.spines.values(), color=3 * [0.4])
            plt.setp([a1.get_xticklines(), a1.get_yticklines()], color=3 * [0.4])

            plt.savefig("img/" + version_name + str(step) + ".png")
            plt.close('all')

        if step % 10000 == 0:
            for idx_point in range(3):
                f = plt.figure(figsize=(10, 10))
                a0 = plt.subplot(1, 1, 1)
                a0.scatter(points[:, 0], points[:, 1], s=32 * 300, facecolors='none', edgecolors='k', linewidth=12,
                           alpha=0.7)
                for kk in range(pp.shape[0]):
                    a0.scatter(pp[kk, 0], pp[kk, 1], s=40 * 100, facecolors='b', alpha=pp[kk, 2])
                a0.set_xlim([points[idx_point, 0] - 0.1, points[idx_point, 0] + 0.1])
                a0.set_ylim([points[idx_point, 1] - 0.1, points[idx_point, 1] + 0.1])
                plt.setp(a0.spines.values(), color=3 * [0.4])
                plt.setp([a0.get_xticklines(), a0.get_yticklines()], color=3 * [0.4])
                a0.axes.get_xaxis().set_ticks([])
                a0.axes.get_yaxis().set_ticks([])
                plt.savefig("zoom/" + version_name + "_zoom{0}.png".format(idx_point))
                plt.close('all')
