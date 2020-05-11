"""
Utility for loading model meterics

@nimrobotics
"""

import numpy as np
import pickle

def load_data(dir):
  with open('./'+dir+'/metrics.pkl', 'rb') as fp:
      metrics = pickle.load(fp)
  with open('./'+dir+'/d_losses.pkl', 'rb') as fp:
      d_losses = pickle.load(fp)
  with open('./'+dir+'/g_losses.pkl', 'rb') as fp:
      g_losses = pickle.load(fp)
      
  metrics = np.array(metrics)
  d_losses = np.array(d_losses)
  g_losses = np.array(g_losses)
  return metrics,d_losses,g_losses
 