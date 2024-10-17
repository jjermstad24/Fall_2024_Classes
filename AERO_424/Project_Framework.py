from scipy.integrate import odeint
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import navpy
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

class Satellite:
    def __init__(self,
                 I = np.array([[400,0,0],[0,300,0],[0,0,200]]),
                 rECI=[0,0,0],
                 vECI=[0,0,0],
                 angles=[0,0,0],
                 rates=[0,0,0],
                 Iw=[0,0,0],
                 Ow=[0,0,0],
                 mu=398600.4418):
        self.I = I
        self.rECI = rECI
        self.vECI = vECI
        self.angles = angles
        self.rates = rates
        self.Iw = Iw
        self.Ow = Ow
        self.mu = mu
        self.state_0 = np.array([rECI[0],rECI[1],rECI[2],
                                 vECI[0],vECI[1],vECI[2],
                                 angles[0],angles[1],angles[2],
                                 rates[0],rates[1],rates[2]])
    
    def from_kepler_elements(self,a,e,i,raan,aop,f,angles=[0,0,0],rates=[0,0,0]):
        self.a = a
        self.e = e
        self.i = i
        self.raan = raan
        self.aop = aop
        self.f = f

        self.T = 2*np.pi*(a**3/self.mu)**0.5

        rPerifocal = a*(1-e**2)/(1+e*np.cos(np.radians(f)))*np.array([np.cos(np.radians(f)),np.sin(np.radians(f)),0])
        vPerifocal = (self.mu/(a*(1-e**2)))**0.5*np.array([-np.sin(np.radians(f)),(e+np.cos(np.radians(f))),0])

        R = Rotation.from_euler("ZXZ", [np.radians(raan), np.radians(i), np.radians(aop)])
        T_P_to_N = R.as_matrix()

        rECI = T_P_to_N@rPerifocal
        vECI = T_P_to_N@vPerifocal

        self.state_0 = np.array([rECI[0],rECI[1],rECI[2],
                                 vECI[0],vECI[1],vECI[2],
                                 angles[0],angles[1],angles[2],
                                 rates[0],rates[1],rates[2]])

    def EOM(self,x,t):
        r = np.linalg.norm(x[:3])
        r_3 = r**3
        I1 = self.I[0,0]
        I2 = self.I[1,1]
        I3 = self.I[2,2]

        L = np.zeros(3)
        if self.enable_torque:
            L += 3*self.mu/r**5*(np.cross(x[:3],np.dot(self.I,x[:3])))
            
        L += [0,-self.Iw[0]*self.Ow[0]*x[11],self.Iw[0]*self.Ow[0]*x[10]]              # Flywheel about b1
        L += [self.Iw[1]*self.Ow[1]*x[11],0,-self.Iw[1]*self.Ow[1]*x[9]]               # Flywheel about b2
        L += [-self.Iw[2]*self.Ow[2]*x[10],self.Iw[2]*self.Ow[2]*x[9],0]               # Flywheel about b3

        return np.array([x[3],
                        x[4],
                        x[5],
                        -self.mu*x[0]/r_3,
                        -self.mu*x[1]/r_3,
                        -self.mu*x[2]/r_3,
                        x[9],
                        x[10],
                        x[11],
                        (((I2-I3)*x[10]*x[11])+L[0])/I1,
                        (((I3-I2)*x[11]*x[9])+L[1])/I2,
                        (((I2-I2)*x[9]*x[10])+L[2])/I3])
    
    def Propagate(self,start,stop,dt,tol=1e-12,enable_torque=False):
        self.enable_torque = enable_torque
        self.times = np.arange(start,stop,dt)
        start = time.time()
        self.TLE = odeint(self.EOM,self.state_0,self.times,rtol=tol,atol=tol)
        return f"Finished Orbit Propagation in: {time.time()-start} s"
    
    def ECEF_TLE(self):
        rx = self.TLE[:,0]
        ry = self.TLE[:,1]
        rz = self.TLE[:,2]
        vx = self.TLE[:,3]
        vy = self.TLE[:,4]
        vz = self.TLE[:,5]

        w = -7.2921159e-5

        gamma = w*self.times
        cg = np.cos(gamma)
        sg = np.sin(gamma)

        rECEF = np.array(
            [cg*rx-sg*ry,
            sg*rx+cg*ry,
            rz])
        
        vECEF = np.array(
            [cg*vx-sg*vy+ry*w,
            sg*vx+cg*vy-rx*w,
            vz])
        
        return rECEF,vECEF