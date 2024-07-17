#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:56:28 2024

@author: shin
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy
from astropy.cosmology import Planck15
import astropy.cosmology as cosmo
import astropy.units as u
from scipy.optimize import minimize
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
from scipy import integrate
import pandas as pd
H0 =  73.8
Om = 0.3
Ok = 0.0
w = -1.0
my_cosmo = cosmo.wCDM(H0=H0, Om0=Om, Ode0=1.0-Om-Ok)


c=300000
q0=-0.55
j0=1
s0=-0.35
l0=3.11
E0=1
E1=1+q0
E2=-q0**2+j0
E3=3*q0**2+3*q0**3-j0*(4*q0+3)-s0
E4=-4*j0**2+25*j0*q0**2+32*j0*q0+12*j0+l0-15*q0**4-24*q0**3-12*q0**2+7*s0*q0+8*s0



def to_cosmographic(H0, Om, Ok, w):
    q0 = 1.0/2.0 *(1.0-Ok-3.0*(-1.0+Om+Ok)*w)
    j0 = 1.0/2.0 * (2.0 * Om + (1.0 - Om - Ok)*(2+9*w*(1+w)))
    s0 = 1.0/4.0 * (2*(7-Ok)*(Ok-1.0)+3*(5*Ok-27)*(1-Om-Ok)*w+9*(1-Om-Ok)*(-16+4*Ok+3*Om)*w**2 + 27.0*(-3.0+Ok+Om)*(1-Ok-Om)*w**3)
    return H0, q0, j0, s0


def dL_EIS0(z,E0):
    dL=np.full(z.shape,np.nan)
    def func(z,E0):
        y=1/(E0)
        return y

    for  i,z_val in  np.ndenumerate(z):
        #print(z_val,integrate.quad(func,0,z_val,args=(E0,E1,E2,E3)))
        dL[i]=(1+z_val)*integrate.quad(func,0,z_val,args=(E0),limit=100)[0]
    return (c/H0)*dL

def dL_EIS1(z,E0,E1):
    dL=np.full(z.shape,np.nan)
    def func(z,E0,E1):
        y=1/(E0+E1*z)
        return y

    for  i,z_val in  np.ndenumerate(z):
        #print(z_val,integrate.quad(func,0,z_val,args=(E0,E1,E2,E3)))
        dL[i]=(1+z_val)*integrate.quad(func,0,z_val,args=(E0,E1),limit=100)[0]
    return (c/H0)*dL

def dL_EIS2(z,E0,E1,E2):
    dL=np.full(z.shape,np.nan)
    zlow=0.03
    
    def func(z,E0,E1,E2):
        if z_val<=zlow:
            y=1/(E0+E1*z)
        else: 
            y=1/(E0+E1*z+(E2*z**2)/2)
        return y

    for  i,z_val in  np.ndenumerate(z):
        #print(z_val,integrate.quad(func,0,z_val,args=(E0,E1,E2,E3)))
        dL[i]=(1+z_val)*integrate.quad(func,0,z_val,args=(E0,E1,E2),limit=100)[0]
    return (c/H0)*dL

def dL_EIS3(z,E0,E1,E2,E3):
    dL=np.full(z.shape,np.nan)
    zlow=0.03
    zmid=0.5
    def func(z,E0,E1,E2,E3):
        if z_val<=zlow:
            y=1/(E0+E1*z)
        elif zlow<z_val<=zmid:
            y=1/(E0+E1*z+(E2*z**2)/2)
        else:
            y=1/(E0+E1*z+(E2*z**2)/2+(E3*z**3)/6)
        return y

    for  i,z_val in  np.ndenumerate(z):
        #print(z_val,integrate.quad(func,0,z_val,args=(E0,E1,E2,E3)))
        dL[i]=(1+z_val)*integrate.quad(func,0,z_val,args=(E0,E1,E2,E3),limit=100)[0]
    return (c/H0)*dL

def dL_EIS4(z,E0,E1,E2,E3,E4):
    dL=np.full(z.shape,np.nan)
    zlow=0.03
    zmid=0.5
    zhigh=0.9
    def func(z,E0,E1,E2,E3,E4):
        if z_val<=zlow:
            y=1/(E0+E1*z)
        elif zlow<z_val<=zmid:
            y=1/(E0+E1*z+(E2*z**2)/2)
        elif zmid<z_val<=zhigh:
            y=1/(E0+E1*z+(E2*z**2)/2+(E3*z**3)/6)
        else:
            y = 1 / (E0 + E1 * z + (E2 * z**2) / 2 + (E3 * z**3) / 6+ (E4 * z**4) / 24)
        return y

    for  i,z_val in  np.ndenumerate(z):
        #print(z_val,integrate.quad(func,0,z_val,args=(E0,E1,E2,E3)))
        dL[i]=(1+z_val)*integrate.quad(func,0,z_val,args=(E0,E1,E2,E3,E4),limit=100)[0]
    return (c/H0)*dL



func_array=[eval(f'dL_EIS{i}') for i in range(0,5)]

zs=np.linspace(0,2,100)
plt.plot(zs, my_cosmo.luminosity_distance(zs).to(u.Mpc).value)
plt.plot(zs,dL_EIS0(zs,E0))
plt.plot(zs,dL_EIS1(zs,E0,E1),color='orange',label='$\mathcal{O}$(2)')
plt.plot(zs,dL_EIS2(zs,E0,E1,E2),color='g',ls='-.',label='$\mathcal{O}$(3)')
plt.plot(zs,dL_EIS3(zs,E0,E1,E2,E3),color='r',ls=":",label='$\mathcal{O}$(4)')
plt.plot(zs,dL_EIS4(zs,E0,E1,E2,E3,E4),color='k',ls='--',label='$\mathcal{O}$(5)')
plt.legend()
plt.xlabel('z')
plt.ylabel('dL')
plt.show()
def chi_square_eis(theta,z,data,func):
    #print(theta)
    #print(func)
    diff = func(z, *theta) - data
    #print(func.__name__,f'{theta}',np.sum(diff*diff))
    return np.sum(diff*diff)



def to_statefinders0(theta):
   
   
    E0_fin=theta[0]
    theta_final=[E0_fin]
    
    return theta_final
def to_statefinders1(theta):

    E0_fin=theta[0]
    E1_fin=theta[1]
    q0=E1_fin - 1
    theta_final=[q0,E0_fin]
    return theta_final

def to_statefinders2(theta):

    E0_fin=theta[0]
    E1_fin=theta[1]
    E2_fin=theta[2]
    q0=E1_fin - 1
    j0=E1_fin**2 - 2*E1_fin + E2_fin + 1

    theta_final=[q0,j0,E0_fin]


    return theta_final

def to_statefinders3(theta):
    E0=theta[0]
    E1=theta[1];
    E2=theta[2]
    q0=E1 - 1
    j0=E1**2 - 2*E1 + E2 + 1
    theta_final=[q0,j0]

    E3=theta[3]
    s0=-4*E1**3 + 12*E1**2 - 4*E1*E2 - 3*E1 + E2 - E3 - 5
    theta_final.append(s0)
    theta_final.append(E0)


    return theta_final

def to_statefinders4(theta):

    E0=theta[0]
    E1=theta[1];
    E2=theta[2]
    q0=E1 - 1
    j0=E1**2 - 2*E1 + E2 + 1
    theta_final=[q0,j0]

    E3=theta[3]
    s0=-4*E1**3 + 12*E1**2 - 4*E1*E2 - 3*E1 + E2 - E3 - 5
    theta_final.append(s0)

    E4=theta[4]
    l0=22*E1**4 - 64*E1**3 + 11*E1**2*E2 - 3*E1**2 - E1*E2 + 7*E1*E3 + 38*E1 + 4*E2**2 + 2*E2 + E3 + E4 + 7
    theta_final.append(l0)

    theta_final.append(E0)
    return theta_final

N=10
ZMAX=2
z_max_array=np.linspace(0.1,ZMAX,N)

X,Y=np.meshgrid(func_array,z_max_array)
x1=np.reshape(X,(1,5*N))[0]
y1=np.reshape(Y,(1,5*N))[0]
res=[]
stat=[]
fun=[]
priors = {
    'dL_EIS0': np.array([E0])+np.random.normal(0,0.5,1),
    'dL_EIS1': np.array([ E0,E1])+np.random.normal(0,0.5,2),
    'dL_EIS2': np.array([ E0, E1,E2])+np.random.normal(0,0.5,3),
    'dL_EIS3': np.array([ E0, E1, E2,E3])+np.random.normal(0,0.5,4),
    'dL_EIS4': np.array([ E0, E1, E2, E3,E4])+np.random.normal(0,0.5,5)
    }

sf = {
    'dL_EIS0': 'to_statefinders0' ,
    'dL_EIS1': 'to_statefinders1',
    'dL_EIS2': 'to_statefinders2',
    'dL_EIS3': 'to_statefinders3',
    'dL_EIS4': 'to_statefinders4'
    }
newdict={}
for p in priors:
    newdict[p]=eval(sf[p])(priors[p])
for func,z_max in list(zip(x1,y1)):
    zs = np.linspace(0.001, z_max, 100)
    func_name = func.__name__
    zs = zs[1:] # We don't want z = 0
    initial_priors = priors[func_name]
    dls_true = my_cosmo.luminosity_distance(zs).to(u.Mpc).value
    print(func_name,z_max)
    
    result = minimize(chi_square_eis, initial_priors,args=(zs, dls_true,eval(func_name)), method='L-BFGS-B', options={'maxiter':950000},tol=10**(-14))
    res.append(eval(sf[func_name])(result.x))
    stat.append(result.success)
    fun.append(result.fun)
  

'''
zs_temp=np.linspace(0.001,0.4,100)
dls_temp = my_cosmo.luminosity_distance(zs_temp).to(u.Mpc).value

#for met in ['L-BFGS-B','trust-constr','Powell','Nelder-Mead','COBYLA']:
for met in ['L-BFGS-B',]:
    result_temp = minimize(chi_square_eis, [1,2,1],args=(zs_temp, dls_temp,dL_EIS2), method='trust-constr', options={'maxiter':950000},tol=10**(-22))
    print(met,round(result_temp.fun,7))
print('zmax is 0.1')
to_statefinders2(result_temp.x)



chi_square_eis(np.array([ 0.9848337 ,  0.44081595,  0.72794931, 10.        , 74.99193818]),zs_temp,dls_temp,dL_EIS3)


chi_square_eis(np.array([ 0.9848337 ,  0.44081595,  0.72794931, 10.        , 74.99193818]),zs_temp,dls_temp,dL_EIS3)

chi_square_eis(np.array([ 1.0710414 ,  0.47933786,  0.79309963, 10.        , 68.95593013]),zs_temp,dls_temp,dL_EIS3)
chi_square_eis(np.array([ 0.73965773,  0.26637225,  1.96793756, 10.        , 99.98308209]),zs_temp,dls_temp,dL_EIS3)
chi_square_eis(np.array([ 6.40027809,  2.86560759,  4.71313724, 10.        , 11.53925515]),zs_temp,dls_temp,dL_EIS3)


chi_square_eis(np.array([ 1.05571268,  0.47249425,  0.78142045,  0.        , 69.95713603]),zs_temp,dls_temp,dL_EIS3)

chi_square_eis(np.array([ 1.05571268,  0.47249425,  0.78142045, 20.        , 69.95713603]),zs_temp,dls_temp,dL_EIS3)


'''
func_name=[i.__name__ for i in np.array(list(zip(x1,y1)))[:,0]]
dct={'func':func_name,'zmax':np.array(list(zip(x1,y1)))[:,1],'Output':res,"success":stat,"function":fun }
df=pd.DataFrame(dct)


H0_ar=[]
for i in df.Output:
    H0_ar.append(i[-1]*H0)
    
q0_ar=[]
ix=0
for j in df.Output :
    if df.func.iloc[ix]!='dL_EIS0':
        q0_ar.append(j[0])
    ix+=1

    
j0_ar=[]
ix=0
for k in df.Output :
    if df.func.iloc[ix]!='dL_EIS0' and df.func.iloc[ix]!='dL_EIS1':
        j0_ar.append(k[1])
    ix+=1




H0_ar=np.reshape(H0_ar,(N,5))
H0_err=abs((73.8-H0_ar)/73.8)*100
H0_err=np.flip(H0_err, axis=0)
q0_ar=np.reshape(q0_ar,(N,4))
q0_err=abs((0.55+q0_ar)/0.55)*100
q0_err=np.flip(q0_err, axis=0)

j0_ar=np.reshape(j0_ar,(N,3))
j0_err=abs((1-j0_ar)/1)*100
j0_err=np.flip(j0_err, axis=0)

##h0plot
plt.figure(figsize=(15,10))
plt.title('$H_0$ variation')
norm = mcolors.LogNorm(vmin=H0_err[H0_err > 0].min(), vmax=H0_err.max())
color_map='YlGnBu'
plt.imshow(H0_err,norm=norm,cmap=color_map)
y_ticks=[i for i in range(int(N))]
y_labels=np.round(np.linspace(ZMAX,0.1,int(N)),1)
x_ticks = [0, 1, 2, 3,4] 
plt.ylabel('$z_{max}$',fontsize=15)
x_labels = ['$\mathcal{O}(1)$', '$\mathcal{O}(2)$', '$\mathcal{O}(3)$', '$\mathcal{O}(4)$','$\mathcal{O}(5)$'] 
plt.xticks(x_ticks, x_labels, rotation ='vertical') 
plt.yticks(y_ticks, y_labels) 
plt.colorbar(label='% deviation')
for i in range(H0_err.shape[0]):
    for j in range(H0_err.shape[1]):
        text = plt.text(j, i, f'{H0_err[i, j]:.1f} %',
                       ha="center", va="center", color="white",path_effects=[path_effects.withStroke(linewidth=1.5, foreground="black")],fontsize=10)

plt.show()




## q0_plot 
plt.figure(figsize=(15,10))
plt.title('$q_0$ variation')
norm = mcolors.LogNorm(vmin=q0_err[q0_err > 0].min(), vmax=q0_err.max())

plt.imshow(q0_err,cmap=color_map,norm=norm)
y_ticks=[i for i in range(int(N))]
y_labels=np.round(np.linspace(ZMAX,0.1,int(N)),1)
x_ticks = [0, 1, 2, 3] 
plt.ylabel('$z_{max}$',fontsize=15)
plt.xlabel('Taylor expansion order')
x_labels = [ '$\mathcal{O}(2)$', '$\mathcal{O}(3)$', '$\mathcal{O}(4)$','$\mathcal{O}(5)$'] 
plt.xticks(x_ticks, x_labels, rotation ='vertical') 
plt.yticks(y_ticks, y_labels) 
for i in range(q0_err.shape[0]):
    for j in range(q0_err.shape[1]):
        text = plt.text(j, i, f'{q0_err[i, j]:.1f} %',
                       ha="center", va="center", color="white",path_effects=[path_effects.withStroke(linewidth=1.5, foreground="black")],fontsize=10)

#v = np.linspace(0, 100, 10, endpoint=True)

plt.colorbar(label='% deviation')
plt.show()


## j0_plot 
plt.figure(figsize=(15,10))
plt.title('$j_0$ variation')
norm = mcolors.LogNorm(vmin=j0_err[j0_err > 0].min(), vmax=j0_err.max())

plt.imshow(j0_err,cmap=color_map,norm=norm)
y_ticks=[i for i in range(int(N))]
y_labels=np.round(np.linspace(ZMAX,0.1,int(N)),1)
x_ticks = [0, 1, 2] 
plt.ylabel('$z_{max}$',fontsize=15)
plt.xlabel('Taylor expansion order')
x_labels = [  '$\mathcal{O}(3)$', '$\mathcal{O}(4)$','$\mathcal{O}(5)$'] 
plt.xticks(x_ticks, x_labels, rotation ='vertical') 
plt.yticks(y_ticks, y_labels) 
for i in range(j0_err.shape[0]):
    for j in range(j0_err.shape[1]):
        text = plt.text(j, i, f'{j0_err[i, j]:.1f} %',
                       ha="center", va="center", color="white",path_effects=[path_effects.withStroke(linewidth=1.5, foreground="black")],fontsize=10)

#v = np.linspace(0, 100, 10, endpoint=True)

plt.colorbar(label='% deviation')
plt.show()

