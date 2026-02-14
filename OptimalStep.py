"""
Définition du pas optimal pour le Perceptron selon la théorie.
"""


import numpy as np


def puissance(A,max_iter=100, tol=1e-3):
  n = A.shape[0]

  x = np.random.rand(n,1)
  x = x/ np.linalg.norm(x)


  vp_old = 0

  for k in range(max_iter):
    y = A@x
    norm_y = np.linalg.norm(y)
    x = y/norm_y

    vp = x.T @ (A @ x)
    vp = vp.item()

    if abs(vp-vp_old) < tol:
      print(f"Convergence atteinte en {k+1} itérations")
      return vp

    vp_old = vp

  return vp
