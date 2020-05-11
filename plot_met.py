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

  N = mmd.shape[0]

  # plt.plot( np.array(range(0,30)) ,emd)
  plt.plot( np.array(range(0,N)) ,mmd)
  plt.plot( np.array(range(0,N)) ,knn_recall)
  plt.plot( np.array(range(0,N)) ,knn_precision)
  plt.plot( np.array(range(0,N)) ,knn_acc_t)
  plt.plot( np.array(range(0,N)) ,knn_acc_f)
  plt.plot( np.array(range(0,N)) ,knn_acc)
  plt.show()

  N = d_losses.shape[0]
  plt.plot(np.array(range(0,N)) ,d_losses, label="D loss")
  N = g_losses.shape[0]
  plt.plot(np.array(range(0,N)) ,g_losses, label="G loss")
  plt.legend()
  plt.xlabel("Iteration")
  plt.show()