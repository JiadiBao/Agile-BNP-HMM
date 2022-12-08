# -*- coding: utf-8 -*-
"""
author: Jiadi
Date: 2022.12.4
CAVI module for agile BNP-HMM
"""

import numpy as np
from scipy.special import digamma

def inv0(X):#矩阵求逆
    try:
        Xm1 = np.linalg.linalg.inv(X)
        return Xm1
    except IndexError:
        return 1/float(X)

class agile_BNP_HMM(object):
    '''
    PE task
    X: data_batch
    K: truncate level
    agile : weather using agile hyperparametger
    kappa : the value of hyper parameter
    '''

    def __init__(self,X,K,Z=None,agile=True, kappa=0):
        self.X = X  # data(N,XDim)
        self.N, self.XDim = self.X.shape
        self._K = K #truncate level
        self.kappa = kappa#跳转系数
        self._hyperparams = {'beta0':(1e-40), 
                        'v0':self.XDim+4, 
                        'W0':(1e-1)*np.eye(self.XDim), 
                        'alpha':1., #concentration parameter of DPMM
                        'alpha_pi':1.,#concentration parameter of initial state
                        'alpha_a':20.,#concentration parameter of transition matrix
                    }
        self.thre = 1e-4
        self.expk = []
        self.agile = agile

    def init_q_param(self):
        '''
        variational distribution initialization
        '''
        self.alpha_pi = self._hyperparams['alpha_pi'] 
        self.alpha_a  = self._hyperparams['alpha_a'] 
        self.alpha = self._hyperparams['alpha'] 

        self.m0 = np.zeros(self.XDim)
        self._W = []
        for k in range(self._K): 
            self._W.append(self._hyperparams['W0'])

        self.exp_z = np.array([np.random.dirichlet(np.ones(self._K)) for _ in range(self.N)])
        self.exp_s = np.array([np.random.uniform(0,100,(self._K,self._K)) for _ in range(self.N)])
        for n in range(self.N): 
            self.exp_s[n,:,:] = self.exp_s[n,:,:] / self.exp_s[n,:,:].sum()

        self.gamma0, self.gamma1 = np.ones(self._K), np.ones(self._K)
        self.tau_pi0, self.tau_pi1 = np.ones(self._K), np.ones(self._K)
        self.tau_a0, self.tau_a1 = np.ones((self._K, self._K)), np.ones((self._K, self._K))
  
    def mixture_fit(self):
        '''
        DPMM混合模型聚类
        结果是self.exp_z,是类别可能性矩阵，有每个点的类别标签的可能性
        每个集群的均值 self._m
        每个集群的方差 self.expC()
        '''
        itr = 0
        diff,prev_ln_obs_lik = 1,np.zeros((self.N,self._K))
        

        while (itr<5) or (itr<200 and diff>self.thre):           
            #---------------
            # M-step:
            #---------------
            
            self.update_V()#Blei Equ.18 Equ.19
            self.update_m_sigma()
            
            #---------------
            # E-step:
            #---------------
            
            ln_obs_lik = self.loglik()#Bishop Equ 10.64 10.65
            self.ln_obs_lik = ln_obs_lik
            exp_ln_pi = self.eV()#Bishop Equ 10.66
            self.mixEZ(ln_obs_lik, exp_ln_pi) #Bishop Equ 10.46 /Blei Equ 22
            
            diff = abs(ln_obs_lik - prev_ln_obs_lik).sum()/float(self.X.shape[0]*self._K) #average difference in previous expected value of transition matrix
            prev_ln_obs_lik = ln_obs_lik.copy()
            
            # print('itr,diff',itr,diff)

            #next iteration:
            itr+=1
            
            #calc state number
            del_index = np.where(~self.exp_z.any(axis=0))[0]
            expz = np.delete(self.exp_z, del_index, axis=1)
            _, expk = expz.shape
            self.expk.append(expk)

    def HMM_fit(self):
        '''
        agile BNP-HMM inference
        '''
        itr = 0
        diff,prev_ln_obs_lik = 1,np.zeros((self.N,self._K)) 
        while (itr<3) or (itr<250 and diff>self.thre):            
            del_index = np.where(~self.exp_z.any(axis=0))[0]
            expz = np.delete(self.exp_z, del_index, axis=1)
            _, expk = expz.shape
            self.expk.append(expk)        
            #---------------
            # M-step:
            #---------------
            
            self.mPi() # parper eq 31,32
            self.mA()# paper eq 33,34
            
            #---------------
            # E-step:
            #---------------
            self.update_m_sigma()# paper eq 35-38            
            ln_obs_lik = self.loglik()#Bishop Equ 10.64
            self.ln_obs_lik = ln_obs_lik
            exp_ln_pi = self.ePi()#Bishop Equ 10.65
            exp_ln_a = self.eA()#Bishop Equ 10.66
            ln_alpha_exp_z = self.eFowardsZ(exp_ln_pi, exp_ln_a, ln_obs_lik) #forward algorithm
            ln_beta_exp_z = self.eBackwardsZ(exp_ln_pi, exp_ln_a, ln_obs_lik) #backward algorithm
            self.eZ(ln_alpha_exp_z, ln_beta_exp_z) #fordward-backward algorithm
            self.eS(exp_ln_a, ln_alpha_exp_z, ln_beta_exp_z, ln_obs_lik) 
            
            diff = abs(ln_obs_lik - prev_ln_obs_lik).sum() / float(self.N*self._K)
            prev_ln_obs_lik = ln_obs_lik.copy()
            
            itr+=1

        
        self.exp_pi = self.expPi()
        self.exp_a = self.expA()   
    

    def del_irr(self,threshold = None):
        '''
        delete the parameter cluster that is irrelevent
        '''
        if threshold is None:
            del_index = np.where(~self.exp_z.any(axis=0))[0]
        else:
            allocate = list(np.argmax(self.exp_z,axis=1))
            del_index=[]
            frequencies = np.zeros(self._K)
            for i in range(self._K):
                frequencies[i] = allocate.count(i)
                if allocate.count(i) == 0:
                    del_index.append(i)
                else:
                    frequency = float(allocate.count(i)/len(allocate))
                    if frequency < threshold:
                        del_index.append(i)
        del_row = np.delete(self.exp_a, del_index, axis=0)
        exp_a = np.delete(del_row, del_index, axis=1)
        _m = np.delete(self._m, del_index)
        exp_z = np.delete(self.exp_z, del_index, axis=1)
        return _m,exp_z,exp_a
    
    def mPi(self):
        K = self._K
        for k in range(K):
            self.tau_pi0[k] = self.alpha_pi + self.exp_z[0,k+1:].sum()
            self.tau_pi1[k] = 1. + self.exp_z[0,k]

    def mA(self):
        K = self._K
        for i in range(K):
            for j in range(K):
                if i == j and self.agile:
                    self.tau_a0[i,j] = self.alpha_a + self.exp_s[:,i,j+1:].sum() + self.kappa * (self.alpha_a + self.exp_s[:,i,j+1:].sum())  
                    self.tau_a1[i,j] = 1. + self.exp_s[:,i,j].sum() 
                else:
                    self.tau_a0[i,j] = self.alpha_a + self.exp_s[:,i,j+1:].sum() 
                    self.tau_a1[i,j] = 1. + self.exp_s[:,i,j].sum()
                
    def eA(self):
        K = self._K
        exp_ln_a = np.zeros((K,K))
        acc = digamma(self.tau_a0) - digamma(self.tau_a0 + self.tau_a1)
        for i in range(K):
            for j in range(K):
                exp_ln_a[i,j] = digamma(self.tau_a1[i,j]) - digamma(self.tau_a0[i,j] + self.tau_a1[i,j]) + acc[i,:j].sum()
        return exp_ln_a
    
    def eFowardsZ(self,exp_ln_pi,exp_ln_a,ln_obs_lik):
        ln_alpha_exp_z = np.zeros((self.N,self._K)) - np.inf
        ln_alpha_exp_z[0,:] = exp_ln_pi + ln_obs_lik[0,:]
        for n in range(1,self.N):
            for i in range(self._K):
                ln_alpha_exp_z[n,:] = np.logaddexp(ln_alpha_exp_z[n,:], ln_alpha_exp_z[n-1,i]+ exp_ln_a[i,:] + ln_obs_lik[n,:])
        return ln_alpha_exp_z 
    
    def eBackwardsZ(self,exp_ln_pi,exp_ln_a,ln_obs_lik):
        N = self.N
        ln_beta_exp_z = np.zeros((self.N,self._K)) - np.inf
        ln_beta_exp_z[N-1,:] = np.zeros(self._K)
        for n in range(N-2,-1,-1):
            for j in range(self._K): 
                ln_beta_exp_z[n,:] = np.logaddexp(ln_beta_exp_z[n,:], ln_beta_exp_z[n+1,j] + exp_ln_a[:,j] + ln_obs_lik[n+1,j])
        return ln_beta_exp_z

    def eZ(self, ln_alpha_exp_z, ln_beta_exp_z):
        ln_exp_z = ln_alpha_exp_z + ln_beta_exp_z
        ln_exp_z -= np.reshape(ln_exp_z.max(axis=1), (self.N,1))
        self.exp_z = np.exp(ln_exp_z) / np.reshape(np.exp(ln_exp_z).sum(axis=1), (self.N,1))

    def eS(self, exp_ln_a, ln_alpha_exp_z, ln_beta_exp_z, ln_obs_lik):
        K = self._K
        N = self.N
        ln_exp_s = np.zeros((N-1,K,K))
        exp_s = np.zeros((N-1,K,K))
        for n in range(N-1):
            for i in range(K):
                ln_exp_s[n,i,:] = ln_alpha_exp_z[n,i] + ln_beta_exp_z[n+1,:] + ln_obs_lik[n+1,:]  + exp_ln_a[i,:]
            ln_exp_s[n,:,:] -= ln_exp_s[n,:,:].max()
            exp_s[n,:,:] = np.exp(ln_exp_s[n,:,:]) / np.exp(ln_exp_s[n,:,:]).sum()
        self.exp_s = exp_s

    def expPi(self):
        K = self._K
        exp_pi = np.zeros((1,K))
        acc = self.tau_pi0 / (self.tau_pi0 + self.tau_pi1)
        for k in range(K): 
            exp_pi[0,k] = (acc[:k].prod()*self.tau_pi1[k]) / (self.tau_pi0[k] + self.tau_pi1[k])
        return exp_pi
    
    def expA(self):
        K = self._K
        exp_a = np.zeros((K,K))
        acc = self.tau_a0/(self.tau_a0+self.tau_a1)
        for i in range(K):
            for j in range(K):
                exp_a[i,j] = (acc[i,:j].prod()*self.tau_a1[i,j])/(self.tau_a0[i,j]+self.tau_a1[i,j]) 
        return exp_a

    def update_m_sigma(self):
        (N,XDim) = np.shape(self.X)
        (N1,K) = np.shape(self.exp_z)
        
        v0 = self._hyperparams['v0']
        beta0 = self._hyperparams['beta0'] 
        self._expW0 = self._hyperparams['W0']       
        
        NK = self.exp_z.sum(axis=0)
        self._NK = NK
        vk = v0 + NK + 1
        self._vk = vk
        xd, S = self._calc_Xk_Sk()
        self._xd = xd
        self._S = S
        betak = beta0 + NK
        self._betak = betak
        self._m = self.update_m(K,XDim,beta0)
        self._W = self.update_W(K,XDim,beta0)

    def loglik(self):
        '''
        calculate the log likelihood
        '''
        K = self._K
        (N,XDim)=np.shape(self.X)
        exp_diff_mu = self._eDiffMu(XDim,N,K) 
        exp_invc = self._eInvc(XDim, K)
        ln_lik = 0.5*exp_invc - 0.5*exp_diff_mu
        return ln_lik

    def _eInvc(self,XDim,K):
        invc = [None for _ in range(K)]
        for k in range(K):
            dW = np.linalg.linalg.det(self._W[k])
            if dW > 1e-30: 
                ld = np.log(dW)
            else: ld = 0.0
            invc[k] = sum([digamma((self._vk[k]+1-i) / 2.) for i in range(XDim)]) + XDim * np.log(2) + ld
        return np.array(invc)

    def _eDiffMu(self,XDim,N,K):
        Mu = np.zeros((N,K))
        A = XDim / self._betak
        for k in range(K):
            B0 = (self.X - self._m[k,:]).T
            B1 = np.dot(self._W[k], B0)
            l = (B0*B1).sum(axis=0)
            assert np.shape(l)==(N,),np.shape(l)
            Mu[:,k] = A[k] + self._vk[k]*l 
        
        return Mu

    def _calc_Xk_Sk(self):
        (N,XDim) = np.shape(self.X)
        (N1,K) = np.shape(self.exp_z)
        assert N==N1
        xd = np.zeros((K,XDim))
        for k in range(K):
            xd[k,:] = (np.reshape(self.exp_z[:,k],(N,1))*self.X).sum(axis=0)
        for k in range(K):
            if self._NK[k]>0: xd[k,:] = xd[k,:]/self._NK[k]
        
        S = [np.zeros((XDim,XDim)) for _ in range(K)]
        for k in range(K):
            B0 = np.reshape(self.X - xd[k,:], (N,XDim))
            for d0 in range(XDim):
                for d1 in range(XDim):
                    L = B0[:,d0]*B0[:,d1]
                    S[k][d0,d1] += (self.exp_z[:,k]*L).sum()
        for k in range(K):
            if self._NK[k]>0: S[k] = S[k]/self._NK[k]

        return xd, S
    
    def expC(self):
        return np.array([inv0(Wk*vk) for (Wk,vk) in zip(self._W,self._vk)])
    
    def update_W(self,K,XDim,beta0):#Bishop Equ 10.62
        Winv = [None for _ in range(K)]
        for k in range(K): 
            Winv[k]  = self._NK[k]*self._S[k] + inv0(self._expW0)
            Q0 = np.reshape(self._xd[k,:] - self.m0, (XDim,1))
            q = np.dot(Q0,Q0.T)
            Winv[k] += (beta0 * self._NK[k] / (beta0 + self._NK[k]) ) * q
            assert np.shape(q)==(XDim,XDim)
        W = []
        for k in range(K):
            try:
                W.append(inv0(Winv[k]))
            except np.linalg.linalg.LinAlgError:
                raise np.linalg.linalg.LinAlgError()
        return W
    
    def update_m(self,K,XDim,beta0):#Bishop Equ.10.61
        m = np.zeros((K,XDim))
        for k in range(K): m[k,:] = (beta0*self.m0 + self._NK[k]*self._xd[k,:]) / self._betak[k]
        return m  
    
    def update_V(self):               #Blei Equ.18 Equ.19
        for k in range(self._K):
            self.gamma0[k] = self.alpha + self.exp_z[:,k+1:].sum() #Blei Eqn 19
            self.gamma1[k] = 1. + self.exp_z[:,k].sum() #Blei Eqn 18
    
    def ePi(self):
        #Blei Equ22
        exp_ln_pi = np.zeros(self._K)
        acc = digamma(self.tau_pi0) - digamma(self.tau_pi0 + self.tau_pi1)
        for k in range(self._K): 
            exp_ln_pi[k] = digamma(self.tau_pi1[k]) - digamma(self.tau_pi0[k] + self.tau_pi1[k]) + acc[:k].sum()
        return exp_ln_pi 

    def eV(self):
        exp_ln_pi = np.zeros(self._K)
        acc = digamma(self.gamma0) - digamma(self.gamma0 + self.gamma1)
        for k in range(self._K): 
            exp_ln_pi[k] = digamma(self.gamma1[k]) - digamma(self.gamma0[k] + self.gamma1[k]) + acc[:k].sum()
        return exp_ln_pi 

    def mixEZ(self,ln_obs_lik, exp_ln_pi):#Bishop Eqn.10.46
        K = self._K
        N = self.X.shape[0]
        ln_exp_z = np.zeros((N,K))
        for k in range(K):
            ln_exp_z[:,k] = exp_ln_pi[k] + ln_obs_lik[:,k]       
        ln_exp_z -= np.reshape(ln_exp_z.max(axis=1), (N,1))
        self.exp_z = np.exp(ln_exp_z) / np.reshape(np.exp(ln_exp_z).sum(axis=1), (N,1))
    