#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:19:24 2019

@author: tylerlanes
"""

# EKF 
import numpy as np
import scipy.linalg as linalg
from scipy.integrate import solve_ivp
from rf45 import r8_rkf45

class ExtendedKalmanFilter_rk45(object):
    
    """ Implements an extended Kalman filter (EKF). You are responsible for
    setting the various state variables to reasonable values; the defaults
    will  not give you a functional filter.
    You will have to set the following attributes after constructing this
    object for the filter to perform properly. Please note that there are
    various checks in place to ensure that you have made everything the
    'correct' size. However, it is possible to provide incorrectly sized
    arrays such that the linear algebra can not perform an operation.
    It can also fail silently - you can end up with matrices of a size that
    allows the linear algebra to work, but are the wrong shape for the problem
    you are trying to solve.
    Parameters
    ----------
    dim_x : int
        Number of state variables for the Kalman filter. For example, if
        you are tracking the position and velocity of an object in two
        dimensions, dim_x would be 4.
        This is used to set the default size of P, Q, and u
    dim_z : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.
    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        State estimate vector
    P : numpy.array(dim_x, dim_x)
        Covariance matrix
    R : numpy.array(dim_z, dim_z)
        Measurement noise matrix
    Q : numpy.array(dim_x, dim_x)
        Process noise matrix
    H : numpy.array(dim_x, dim_x)
        Measurement function
    ri: numpy.array(dim_z, 1) (pre and post)
        Residual of the update step. Read only.
    z : ndarray
        Last measurement used in update(). Read only.
        """

    def __init__(self, x_0, P_0, dim_x, dim_z, dim_u=0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.xm = np.zeros((dim_x, 1)) # x-minus
        self.x = x_0               # x-plus        
        self.P = P_0               # uncertainty covariance
        self.R = np.eye(dim_z)        # state uncertainty
        self.Q = np.eye(dim_x)        # process uncertainty
        self.y = np.zeros((dim_z, 1)) # residual
        self.ri_pre     = np.zeros((dim_z, 1)) # residual
        self.ri_post    = np.zeros((dim_z, 1)) # residual                
        self.STM        = np.zeros((dim_x,dim_x)) # state transition matrix
        self.sig        = np.zeros((dim_x,1)) #standard deviation

        z = np.array([None]*self.dim_z)   #idk what this is about
        self.z = z

        # bad inits - gonna give ya a bad time
        self.qmag = 0
        self.integ_fcn  = None
        self.Htil_fcn   = None
        self.h_fcn      = None
        self.Q_fcn      = None
        self.time       = 0
        self.xd         = 0
        self.flag       = 1
        
        
    def time_update(self, tspan, args=(),reltol=np.sqrt(np.finfo(np.double).eps), abstol=np.sqrt(np.finfo(np.double).eps)):
        x = self.x
        P = self.P
        s = np.reshape(np.eye(self.dim_x),(self.dim_x**2,1))
#        self.xd = self.integ_fcn(0, np.append(self.x,s))
        xd = self.xd
        flag = self.flag
        
        dt = tspan[1]-tspan[0]
        if self.Q_fcn:
            Qi = self.Q_fcn(dt,self.qmag,args)
        else:
            Qi = self.make_q(dt)
        
        #integrate/propagate to next time step
        t = tspan[0]
        t_out = tspan[1]
        flag=1
        x, xd, t, flag = r8_rkf45 ( self.integ_fcn, self.dim_x, np.append(x,s), xd, t, t_out, reltol, abstol, flag )
        xi = x
        
        stm = np.reshape(xi[self.dim_x:], (self.dim_x,self.dim_x), order='F')
        Pi  = stm @ P @ stm.T + Qi

        #save time propagated state
        self.xm     = xi[:self.dim_x]
        self.xd     = xd
        self.Pm     = Pi
        self.STM    = stm
        self.Q      = Qi
        self.time   = tspan[1]
        self.flag   = flag
            
    def meas_update(self, z, H_args=()):
        xm = self.xm
        if z is None:
            self.P          = self.Pm
            self.x          = self.xm
            self.ri_post    = None
            self.ri_pre     = None
            self.sig        = 3 * np.sqrt(np.diag(self.P))
            return
        
        #compute obs
        sim_meas    = self.h_fcn(xm, *H_args)
        ri_pre      = z - sim_meas #prefit res
        H           = self.Htil_fcn(xm, *H_args) #H jacobian
        hphr        = H @ self.Pm @ H.T + self.R
        K           = self.Pm @ H.T @ linalg.inv(hphr) # Kalman gain
        
        #meas update
        xi      = xm + K @ ri_pre
        i_kh    = np.eye(self.dim_x) - (K @ H)
        Pi      = i_kh @ self.Pm @ i_kh.T + (K @ self.R @ K.T)
        sig     = 3 * np.sqrt(np.diag(Pi))
        #post fit
        sim_meas    = self.h_fcn(xi, *H_args)
        ri_post     = z - sim_meas
        
        #save meas updated state, covar, resid, and meas
        self.x       = xi
        self.P       = Pi
        self.ri_post = ri_post
        self.ri_pre  = ri_pre
        self.sig     = sig
        self.z       = z
    
    #RMS function
    def grabRMS(self, x):
        row,col = x.shape
        RMS = np.zeros([row,1])
        for i in range(row):
            RMS[i] = np.sqrt(np.sum(np.dot(x[i,:],x[i,:]))/col)
        return RMS        
    
    #lame process noise function 
    def make_q(self, dt):
        return self.qmag * np.eye(self.dim_x)
    
    
################################
        
class ExtendedKalmanFilter_sivp(object):
    
    def __init__(self, x_0, P_0, dim_x, dim_z, dim_u=0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.xm = np.zeros((dim_x, 1)) # x-minus
        self.x = x_0               # x-plus        
        self.P = P_0               # uncertainty covariance
        self.R = np.eye(dim_z)        # state uncertainty
        self.Q = np.eye(dim_x)        # process uncertainty
        self.y = np.zeros((dim_z, 1)) # residual
        self.ri_pre     = np.zeros((dim_z, 1)) # residual
        self.ri_post    = np.zeros((dim_z, 1)) # residual                
        self.STM        = np.zeros((dim_x,dim_x)) # state transition matrix
        self.sig        = np.zeros((dim_x,1)) #standard deviation

        z = np.array([None]*self.dim_z)   #idk what this is about
        self.z = z

        # bad inits - gonna give ya a bad time
        self.qmag = 0
        self.integ_fcn  = None
        self.Htil_fcn   = None
        self.h_fcn      = None
        self.Q_fcn      = None
        self.time       = 0
        
        
    def time_update(self, tspan, args=(),reltol=1e-9, abstol=1e-9):
        x = self.x
        P = self.P
        s = np.reshape(np.eye(self.dim_x),(self.dim_x**2,1))

        dt = tspan[1]-tspan[0]
        if self.Q_fcn:
            Qi = self.Q_fcn(dt,self.qmag,args)
        else:
            Qi = self.make_q(dt)
        
        #integrate/propagate to next time step
        r   = solve_ivp(self.integ_fcn, tspan, np.append(x,s), method='RK45', rtol=reltol, atol=abstol)
        xi  = r.y[:,-1]        
        stm = np.reshape(xi[self.dim_x:], (self.dim_x,self.dim_x), order='F')
        Pi  = stm @ P @ stm.T + Qi

        #save time propagated state
        self.xm     = xi[:self.dim_x]
        self.Pm     = Pi
        self.STM    = stm
        self.Q      = Qi
        self.time   = tspan[1]
            
    def meas_update(self, z, H_args=()):
        xm = self.xm
        if z is None:
            self.P          = self.Pm
            self.x          = self.xm
            self.ri_post    = None
            self.ri_pre     = None
            self.sig        = 3 * np.sqrt(np.diag(self.P))
            return
        
        #compute obs
        sim_meas    = self.h_fcn(xm, *H_args)
        ri_pre      = z - sim_meas #prefit res
        H           = self.Htil_fcn(xm, *H_args) #H jacobian
        hphr        = H @ self.Pm @ H.T + self.R
        K           = self.Pm @ H.T @ linalg.inv(hphr) # Kalman gain
        
        #meas update
        xi      = xm + K @ ri_pre
        i_kh    = np.eye(self.dim_x) - (K @ H)
        Pi      = i_kh @ self.Pm @ i_kh.T + (K @ self.R @ K.T)
        sig     = 3 * np.sqrt(np.diag(Pi))
        #post fit
        sim_meas    = self.h_fcn(xi, *H_args)
        ri_post     = z - sim_meas
        
        #save meas updated state, covar, resid, and meas
        self.x       = xi
        self.P       = Pi
        self.ri_post = ri_post
        self.ri_pre  = ri_pre
        self.sig     = sig
        self.z       = z
    
    #RMS function
    def grabRMS(self, x):
        row,col = x.shape
        RMS = np.zeros([row,1])
        for i in range(row):
            RMS[i] = np.sqrt(np.sum(np.dot(x[i,:],x[i,:]))/col)
        return RMS        
    
    #lame process noise function 
    def make_q(self, dt):
        return self.qmag * np.eye(self.dim_x)
        
    
###############################################################################
        
class ClassicKalmanFilter(object):
    
    def __init__(self, x_0, P_0, dim_x, dim_z, dim_u=0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.dxm = np.zeros((dim_x, 1)) # dx-minus
        self.dx  = np.zeros((dim_x, 1)) # dx plus
        self.x = np.reshape(x_0,(dim_x,1),order='F')   # x        
        self.Pm = np.zeros((dim_x,dim_x)) #covar minus
        self.P = P_0               # uncertainty covariance
        self.R = np.eye(dim_z)        # state uncertainty
        self.Q = np.eye(dim_x)        # process uncertainty
        self.y = np.zeros((dim_z, 1)) # residual
        self.ri_pre     = np.zeros((dim_z, 1)) # residual
        self.ri_post    = np.zeros((dim_z, 1)) # residual                
        self.STM        = np.zeros((dim_x,dim_x)) # state transition matrix
        self.sig        = np.zeros((dim_x,1)) #standard deviation

        #z = np.array([None]*self.dim_z)   #idk what this is about
        self.z = np.zeros((dim_z,1))

        # bad inits - gonna give ya a bad time
        self.qmag = 0
        self.integ_fcn  = None
        self.Htil_fcn   = None
        self.h_fcn      = None
        self.Q_fcn      = None
        self.time       = 0
        self.xd         = 0
        
    def time_update(self, tspan, args=(),reltol=1e-12, abstol=1e-12):
        x = self.x
        P = self.P
        s = np.reshape(np.eye(self.dim_x),(self.dim_x**2,1))
        xd = self.xd
        
        dt = tspan[1]-tspan[0]
        if self.Q_fcn:
            Qi = self.Q_fcn(dt,self.qmag,args)
        else:
            Qi = self.make_q(dt)
        
        #integrate/propagate to next time step
        t = tspan[0]
        t_out = tspan[1]
        flag=1
        x, xd, t, flag = r8_rkf45 ( self.integ_fcn, self.dim_x, np.append(x,s), xd, t, t_out, reltol, abstol, flag )
        xi = x
        stm = np.reshape(xi[self.dim_x:], (self.dim_x,self.dim_x), order='F')
        dxm = stm @ self.dx
        Pi  = stm @ P @ stm.T + Qi

        #save time propagated state
        self.dxm    = dxm
        self.x      = np.expand_dims(xi[:self.dim_x],axis=1)
        self.Pm     = Pi
        self.STM    = stm
        self.Q      = Qi
        self.time   = tspan[1]
            
    def meas_update(self, z, H_args=()):
        x = np.squeeze(self.x)
        dxm = self.dxm
        if z is None:
            self.P          = self.Pm
            self.dx          = self.dxm
            self.ri_post    = None
            self.ri_pre     = None
            self.sig        = 3 * np.sqrt(np.diag(self.P))
            return
        
        #compute obs
        sim_meas    = self.h_fcn(x, *H_args)
        ri_pre      = np.reshape(z - np.asarray(sim_meas),(self.dim_z,1)) #prefit res
        H           = self.Htil_fcn(x, *H_args) #H jacobian
        hphr        = H @ self.Pm @ H.T + self.R
        K           = self.Pm @ H.T @ linalg.inv(hphr) # Kalman gain
        
        #meas update
        innov   = ri_pre - H @ dxm
        dxi     = dxm + K @ innov
        i_kh    = np.eye(self.dim_x) - (K @ H)
        Pi      = i_kh @ self.Pm @ i_kh.T + (K @ self.R @ K.T)
        sig     = 3 * np.sqrt(np.diag(Pi))
        #post fit
        ri_post     = ri_pre - H @ dxi
        
        #save meas updated state, covar, resid, and meas
        self.dx      = dxi
        self.P       = Pi
        self.ri_post = ri_post
        self.ri_pre  = ri_pre
        self.sig     = sig
        self.z       = z
    
    #RMS function
    def grabRMS(self, x):
        row,col = x.shape
        RMS = np.zeros([row,1])
        for i in range(row):
            RMS[i] = np.sqrt(np.sum(np.dot(x[i,:],x[i,:]))/col)
        return RMS        
    
    #lame process noise function 
    def make_q(self, dt):
        return self.qmag * np.eye(self.dim_x)
    
    
    
    
    
    
    
###############################################################################    
    
class UnscentedKalmanFilter(object):

    def __init__(self, x_0, P_0, dim_x, dim_z, dim_u=0, sqrt_fn=None):

        self.x = x_0
        self.P = P_0
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.hx = None
        self.fx = None
        self.alpha  = 1
        self.beta   = 2
        self.kappa  = 0
        self.lamb   = self.alpha**2 *(self.kappa+dim_x) - dim_x
        self.gamm   = np.sqrt(self.lamb + dim_x)
        
        if sqrt_fn is None:
            self.msqrt = linalg.sqrtm
        else:
            self.msqrt = sqrt_fn

        # weights for the means and covariances.
        self.Wm0= self.lamb/(dim_x+self.lamb)
        self.Wc0 = self.lamb/(dim_x+self.lamb)+1-self.alpha**2+self.beta
        wvec = np.ones([1,2*dim_x]) * (1/(2*(self.lamb+dim_x)))
        self.Wm = np.insert(wvec, 0, self.Wm0)
        self.Wc = np.insert(wvec, 0, self.Wc0)
        self.ind = 2*dim_x + 1
        
        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update

        self.sigmas_f = np.zeros((self.ind, dim_x))
        self.sigmas_h = np.zeros((self.ind, dim_z))

        self.K = np.zeros((dim_x, dim_z))    # Kalman gain
        self.ri_pre = np.zeros((dim_z))           # residual
        self.z = np.array([[None]*dim_z]).T  # measurement
        self.time = 0
        self.qmag = 0
        
        
    def time_update(self, tspan, args=(), reltol=1e-12, abstol=1e-12):
        x = np.expand_dims(self.x,axis=1)
        P = self.P
        
        dt = tspan[1]-tspan[0]
        if self.Q_fcn:
            Qi = self.Q_fcn(dt,self.qmag, *args)
        else:
            Qi = self.make_q(dt)
            
        X_m = np.hstack((x,x+self.gamm*self.msqrt(P)))
        X_m = np.hstack((X_m,x-self.gamm*self.msqrt(P)))    
        #integrate/propagate to next time step
        x0 = np.reshape(X_m,((self.ind*self.dim_x),),order='F')
        r   = solve_ivp(self.integ_fcn, tspan, x0, method='RK45', rtol=reltol, atol=abstol)
        X  = r.y[:,-1]  
        X = np.reshape(X, (self.dim_x,self.ind))
        
        x = np.zeros([self.dim_x,])
        for j in range(self.ind):
            x += self.Wm[j] * X[:,j]
        
        for j in range(self.ind):
            P += self.Wc[j] * np.outer(X[:,j]-x, X[:,j]-x)
        P += Qi
        x = np.expand_dims(x,axis=1)
        X = np.hstack((x,self.gamm*self.msqrt(P)))
        X = np.hstack((X,self.gamm*self.msqrt(P)))
        
        self.sigmas_f = X  
        self.xm       = x
        self.Pm = P     
        self.time = tspan[1]
        print(self.time)
        
        
    def meas_update(self, z, H_args=()):
        if z is None:
            self.x = self.xm
            self.P = self.Pm
            self.sig = self.sig
            self.ri_pre = None
            self.ri_post = None
            return
        
        x = self.xm
        X = self.sigmas_f
        Y = self.h_fcn(X, *H_args)
        y = np.zeros([self.dim_z,])
        for j in range(self.ind):
            y += self.Wm[j] * Y[:,j]
            
        Pyy = np.zeros([self.dim_z,self.dim_z])
        Pxy = np.zeros([self.dim_x,self.dim_z])        
        for j in range(self.ind):
            Pyy += self.Wc[j] * np.outer(Y[:,j]-y, Y[:,j]-y)
            Pxy += self.Wc[j] * np.outer(X[:,j]-np.squeeze(x), Y[:,j]-y)
        
        Pyy += self.R
        
        K = Pxy @ (linalg.inv(Pyy))
        x += K @ np.expand_dims((z - y),axis=1)
        P = self.Pm - K @ Pyy @ K.T
        sig = 3 * np.sqrt(np.diag(P))
        rpre = z - y
        rpost = z - np.squeeze(self.h_fcn(x, *H_args))
        
        self.x = np.squeeze(x)
        self.P = P
        self.sig = sig
        self.ri_pre = rpre
        self.ri_post = rpost
        self.z = z
 
        
        #RMS function
    def grabRMS(self, x):
        row,col = x.shape
        RMS = np.zeros([row,1])
        for i in range(row):
            RMS[i] = np.sqrt(np.sum(np.dot(x[i,:],x[i,:]))/col)
        return RMS        
    
    #lame process noise function 
    def make_q(self, dt):
        return self.qmag * np.eye(self.dim_x)
    
    
    
    
    
    
    
    