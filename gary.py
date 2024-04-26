import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Useful shorthand definitions
pi = np.pi
sqrt = np.sqrt
exp = np.exp

# Parameters

M0 = 1477                                                  #solarmass #instasun
Q0 = 1.7e5*M0                                                    # equation 8
hbar = 2.6e-70
m = 6.8e-58
qe = 1.45e-41
constant = (qe**4)/(2.0*pi**3*hbar*m**2)
a = pi**2/(15.0*hbar**3)                    # constant for Stefan-Boltzmann law (unnumbered, below 12)
alphaHW = 2.0228
params = (M0, Q0, hbar, m, qe, constant, a, alphaHW)


# Defining the evaporation rate for Q & M

def rq(y):
    return 1 + sqrt(1-max(y,0)) 

def rp(M,y):
    return M*rq(y)

def T(M,y):
    return (hbar/2*pi)*sqrt(1 - max(y,0))/(M*rq(y)**2)

def sigma0(M,y):
    return pi*M**2*(3 + sqrt(9 - 8*max(y,0)))**4/(8*(3 - 2*y + sqrt(9 - 8*max(y,0))))      #equation (13)

def alpha(M,y):
    return a*alphaHW*sigma0(M,y)

def beta(M,y):
    return -alpha(M,y)*T(M,y)**4

def fy(M,y):
    if y>0:
        return -2*constant*y**2*(rq(y)-y)*exp(-M*rq(y)**2/(Q0*sqrt(y)))/(rq(y)**4*(M+2*beta(M,y)*y))
    else:
        return  0

def f(t, x):
    M, y = x   
    if M<0:
        fM = 0
        fY = 0
    else:
        fM = (beta(M,y)+(M/(2*rq(y)))*fy(M,y))/(1-(y/rq(y)))
        fY = fy(M,y) 
    return np.array([fM, fY], np.float64)

#RK4 method:
def rk4_step(g,t,x):
    k1 = h*g(t,x)
    k2 = h*g(t + 0.5*h,x + 0.5*k1)
    k3 = h*g(t + 0.5*h,x + 0.5*k2)
    k4 = h*g(t + h,x + k3)
    return((k1 + 2.0*(k2 + k3) + k4)/6.0) #Retorna posição no inst. seguinte

#RK4 integration
#errorlist = []
#for omega in Omega:
#    x_rk4 = []
#    xrk4 = x0
#    erro = 0
#    for t_rk4 in t_grid:
#        x_rk4.append(xrk4)
#        xrk4 += rk4_step(f,t_rk4,xrk4)
#        erro += (xrk4 - x_ex(t_rk4))**2
#    errorlist.append(((erro)/N)**0.5)


# Inital parameters

Mi = 1.0e4*M0     # initial mass
yi = 0.6       # initial Q^2/M^2
tStop = 1e90
#*(Mi**3)
tStopQ = tStop
#M=10, y=0.6 tStop = 1e72
xi = [Mi, yi]

nn = 100    # number of points inside each ODE piece
tStop_int = 0
Qstopped = False
i=0
Mass_full = np.empty(shape=[0,nn-1])
Charge_full = np.empty(shape=[0,nn-1])
teval_full = np.empty(shape=[0])    


# Call the ODE solver
Mi_int, yi_int = xi
while tStop_int<tStop and i<200:  
    # First I determine the internal tStop
    if f(0,xi)[0]!=0.0: # t=0 can be used as all time-dependence enters implicitly through xi
        if abs(f(0,xi)[0]/Mi_int)<abs(f(0,xi)[1]/yi_int):
            tStop_int = abs(yi_int/f(0,xi)[1])
        else:
            tStop_int = abs(Mi_int/f(0,xi)[0])*1e-2
    else:
        tStop_int = tStop   
    t = [0.0,tStop_int]
 

    #Solving the ODE
    
    psol_int = solve_ivp(f, t, xi, method='RK45', dense_output=True, atol=1.5e-7)
    teval_int = np.linspace(0, tStop_int, nn)
    xsol_int = psol_int.sol(teval_int)
    
    #transforming the answer into somehting nice
    #to get rid of the zeros in time (it was putting the same point twice)   
    
    Mass2 = np.array(xsol_int[0])
    Mass = np.zeros(nn-1)
    m=0
    for m in range(nn-1):
        Mass[m] = Mass2[m+1]
        
    Charge2 = np.array(xsol_int[1])
    Charge = np.zeros(nn-1)
    m=0
    for m in range(nn-1):
        Charge[m] = Charge2[m+1]
        
    teval0 = np.zeros(nn-1)
    m=0
    for m in range(nn-1):
        teval0[m] = teval_int[m+1]

    # Add the new solution to the previous one
    
    Mass_full = np.vstack((Mass_full, Mass))
    Charge_full = np.vstack((Charge_full, Charge))
    if len(teval_full) == 0:
        teval_full = teval0
        tempo = np.zeros(nn-1)
        for l in range(nn-1):
            tempo[l] = teval0[nn-2]
    else:
        teval_full = np.vstack((teval_full,tempo + teval0))
        tempo = np.zeros(nn-1)  
        l=0
        for l in range(nn -1):
            tempo[l] = teval_full[i,nn-2]
    # Now set new initial values
    xi = (xsol_int.T[len(xsol_int.T)-1,0], xsol_int.T[len(xsol_int.T)-1,1])
    if Charge[nn-2] <= 1e-1 and not(Qstopped):
        tStopQ = teval_full[i,nn-2]
        Qstopped = True
    i += 1
    
tplot = np.zeros((nn-1)*len(teval_full))
j=0
i=0
while j < len(Charge_full):
    if i < (len(Charge_full[0])):
        tplot[i + j*(nn-1)] = teval_full[j,i]
        i += 1
    else:
        i = 0
        j += 1


Mplot = np.zeros((nn-1)*len(Mass_full))
j=0
i=0
while j < len(Mass_full):
    if i < len(Mass_full[0]):
        Mplot[i +j*(nn-1)] = Mass_full[j,i]
        i += 1
    else:
        i = 0
        j += 1

Qplot = np.zeros((nn-1)*len(Charge_full))
j=0
i=0
while j < len(Charge_full):
    if i < len(Charge_full[0]):
        Qplot[i +j*(nn-1)] = Charge_full[j,i]
        i += 1
    else:
        i = 0
        j += 1

#Plot results
with plt.rc_context({'figure.facecolor':'white'}):
    fig = plt.figure(1, figsize=(8,8))

    # Plot M as a function of time
    ax1 = fig.add_subplot(311)
    ax1.plot(tplot, Mplot)
    ax1.set_xlabel('time')
    ax1.set_ylabel('M')
    #ax1.set_xlim(0,2.5e71)
    #ax1.set_xlim(0,1e-16)

    # Plot (Q/M)^2 as a function of time
    ax2 = fig.add_subplot(312)
    ax2.plot(tplot, Qplot)
    ax2.set_xlabel('time')
    ax2.set_ylabel('(Q/M)^2')
    ax2.set_ylim(0., 1.)
    ax2.set_xlim(0,tStopQ)

    # Plot (Q/M)^2 vs M
    ax3 = fig.add_subplot(313)
    ax3.plot(Mplot, Qplot, '-')
    ax3.set_xlabel('M')
    ax3.set_ylabel('(Q/M)^2')
    ax3.set_ylim(0., 1.)

    plt.tight_layout()
    plt.show()
    
    # Error:  <ipython-input-48-e99df5ef8d96>:60: RuntimeWarning: overflow encountered in double_scalars
    # alpha = a*alphaHW*sigma0
    
    # TypeError: object of type 'numpy.float64' has no len()

