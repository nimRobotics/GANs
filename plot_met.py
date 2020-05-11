"""
Utility for plotting GAN metrics

@nimrobotics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_data(metrics,d_losses,g_losses):
  emd = metrics[:,0]
  mmd = metrics[:,1]
  knn_recall = metrics[:,2]
  knn_precision = metrics[:,3]
  knn_acc_t = metrics[:,4]
  knn_acc_f = metrics[:,5]
  knn_acc = metrics[:,6]

  print("mmd", np.mean(mmd), np.var(mmd))
  print("knn_acc_t", np.mean(knn_acc_t), np.var(knn_acc_t))
  print("knn_acc_f", np.mean(knn_acc_f), np.var(knn_acc_f))
  print("knn_acc", np.mean(knn_acc), np.var(knn_acc))

  N = mmd.shape[0]

  # plt.plot( np.array(range(0,30)) ,emd)
  plt.plot( np.array(range(0,N)) ,mmd, label="mmd")
  # plt.plot( np.array(range(0,N)) ,knn_recall,label="")
  # plt.plot( np.array(range(0,N)) ,knn_precision)
  plt.plot( np.array(range(0,N)) ,knn_acc_t, label="1 NN Accuracy (true)")
  plt.plot( np.array(range(0,N)) ,knn_acc_f, label="1 NN Accuracy (fake)")
  plt.plot( np.array(range(0,N)) ,knn_acc, label="1 NN Accuracy")
  plt.legend()
  plt.xlabel("Iterations")
  plt.show()

  N = d_losses.shape[0]
  plt.plot(np.array(range(0,N)) ,d_losses, label="D loss")
  N = g_losses.shape[0]
  plt.plot(np.array(range(0,N)) ,g_losses, label="G loss")
  plt.legend()
  plt.xlabel("Iterations")
  plt.ylabel("Loss")
  plt.show()