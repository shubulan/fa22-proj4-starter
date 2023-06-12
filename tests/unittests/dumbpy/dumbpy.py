import numpy as np
import numc as nc
class Matrix:
  def __init__(self, *args, **kwargs):
    if not args:
      self.data = None
      self.shape = None
      return
    if kwargs:
      x = nc.Matrix(*args, **kwargs)
    else:
      x = nc.Matrix(*args)
    self.data = np.zeros(x.shape)
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        self.data[i, j] = x[i][j]
  
    self.shape = self.data.shape

  def __str__(self):
    return str(self.data)
  def __repr__(self):
    return repr(self.data)

  def __abs__(self):
    abs(self.data)
    return self
  
  def __add__(self, other):
    ret = Matrix()
    ret.data = self.data + other.data
    ret.update_shape()
    return ret
  
  def update_shape(self):
    self.shape = self.data.shape

  def get(self, i, j):
    return self.data[i, j]
  
  
