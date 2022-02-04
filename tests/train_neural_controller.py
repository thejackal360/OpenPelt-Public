import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

from neural_nets import MLP
from torch import nn


if __name__ == '__main__':
    batch_size = 1
    epochs = 25

    v, tc = np.load("results/op_point_voltage_volt_tc_characterization.npy").astype('f')
    v, th = np.load("results/op_point_voltage_volt_th_characterization.npy").astype('f')
    size = min(len(tc), len(th))
    t_ref = np.linspace(0, 10, size).astype('f')

    X = np.vstack([tc[:size], th[:size], t_ref]).T
    y = v.T
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=13)

    nc = neural_controller(T_ref=t_ref[0],
                           input_units=3,
                           hidden_units=6,
                           bias=True,
                           lrate=0.0005)

    train_patterns = int(X_train.shape[0] / batch_size)

    loss_track, acc_track = [], []
    for e in range(epochs):
        for i in range(X_train.shape[0]):
            x = torch.from_numpy(np.array([X_train[i, 1],
                                           X_train[i, 0],
                                           X_train[i, 2]]).astype('f'))
            self.t_ref = torch.tensor([T_ref], requires_grad=True)
            self.z = torch.tensor([T_hot], requires_grad=True)

            self.optimizer.zero_grad()
            yc = self.net(x)
        if e % 1 == 0:
            acc = 0.0
            for j in range(X_test.shape[0]):
                pred = nc.controller(X_test[j, 1], X_test[j, 0], X_test[j, 2])
                if np.abs(pred - y_test[j]) <= 0.1:
                    acc += 1.0
            acc /= X_test.shape[0]
            loss = np.mean(nc.loss_)
            loss_track.append(loss)
            acc_track.append(acc)
            print("[Epoch %d, Loss %f, Accuracy: %f]" % (e, nc.loss_[-1], acc))

    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plt.plot(loss_track)
    plt.subplot(122)
    plt.plot(acc_track)
    plt.show()
