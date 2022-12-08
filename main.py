# -*- coding: utf-8 -*-
"""
author: Jiadi
Date: 2022.12.4
main function of agile BNP-HMM
The purpose of these codes are for supporting the manuscript submission.
"""

import numpy as np
import matplotlib.pyplot as plt
from HMM_BNP_func import agile_BNP_HMM

D = np.load('demo_dataset.npy',allow_pickle = True)
X = np.array([D[:,0]]).T
Z = np.array([D[:,1]]).T
bkps_truth = [150,300,450]
bank = [(210/255,57/255,24/255),(229/255,168/255,75/255),(255/255,238/255,111/255),(166/255,64/255,54/255),(93/255,163/255,157/255),(73/255,148/255,196/255),(21/255,29/255,41/255)]# Chinese style color

#kappa=0
model = agile_BNP_HMM(X, K=15, Z=Z, agile=True, kappa = 0)
model.init_q_param()#initialize all q distributions
model.HMM_fit()# model fitting
mu,Z,A = model.del_irr()# delete irrelevent parameter clusters


plt.figure(dpi=150,figsize=(8,5))
plt.xlabel("PRI index")
plt.ylabel("PRI value")
plt.title("Staggered PRI demo (kappa = 0)")

for i, x in enumerate(X):
    z_n = np.argmax(Z[i])
    plt.plot(i,x,"*-",c=bank[z_n])

# kappa = 1
model = agile_BNP_HMM(X, K=15, Z=Z, agile=True, kappa = 1)
model.init_q_param()#initialize all q distributions
model.HMM_fit()# model fitting
mu,Z,A = model.del_irr()# delete irrelevent parameter clusters


plt.figure(dpi=150,figsize=(8,5))
plt.xlabel("PRI index")
plt.ylabel("PRI value")
plt.title("Staggered PRI demo (kappa = 1)")

for i, x in enumerate(X):
    z_n = np.argmax(Z[i])
    plt.plot(i,x,"*-",c=bank[z_n])
plt.show()
