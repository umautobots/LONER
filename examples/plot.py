import numpy as np
import matplotlib.pyplot as plt

data = np.load("data.npy")
gt = np.load("gt.npy")

xs = data[0,:]
ys = data[1,:]
zs = data[2,:]

xs_gt = gt[0,:]
ys_gt = gt[1,:]
zs_gt = gt[2,:]


fig = plt.figure()

ax=fig.add_subplot(projection='3d')
plt.plot(xs, ys, zs, color='r', label="Estimated")
plt.plot(xs_gt, ys_gt, zs_gt, color='g', label="Ground Truth")

plt.show()