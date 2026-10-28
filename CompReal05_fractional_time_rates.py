"""Computational reality 05, linear rheology with fractional time rates"""
__author__ = "B. Emek Abali"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

import time as comp_time
start_time = comp_time.time()

from dolfin import *
from ufl_legacy import indices
#import scipy
from scipy.special import factorial, gamma
import os, shutil
#import numpy as np
set_log_level(50)
pwd='results/CR05/'


xlength =0.100 #[m]
ylength =0.025 #[m]
mesh = RectangleMesh(Point(0,0),Point(xlength,ylength),20,5)
Dim = mesh.geometric_dimension()

class PeriodicBoundary(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0],0) and on_boundary
	def map(self, y, x):
		#this maps right side x[0]=xlength to the left
		x[0] = y[0]-xlength
		x[1] = y[1]

PeriodicBC = PeriodicBoundary()

Space = VectorFunctionSpace(mesh,'CG',1, constrained_domain=PeriodicBC)
TensorSpace = TensorFunctionSpace(mesh,'CG',1, constrained_domain=PeriodicBC)

delu = TestFunction(Space)
du = TrialFunction(Space)
u = Function(Space)
u0 = Function(Space)
u00 = Function(Space)

print('initializing,time')
t_start=0.0
t_end = 10.0
i, j, k, l = indices(4)


#setting initial conditions
initialcond = Expression(('0','0'),degree=1)
u0.interpolate(initialcond)
u00.assign(u0)

#Defining boundary conditions
left = CompiledSubDomain('near(x[0],0) && on_boundary')
right = CompiledSubDomain('near(x[0],l) && on_boundary',l=xlength)
bottom = CompiledSubDomain('near(x[1],0) && on_boundary')
top = CompiledSubDomain('near(x[1],l) && on_boundary',l=ylength)

cells = MeshFunction('size_t', mesh, Dim)
facets = MeshFunction('size_t', mesh, Dim-1)
dA = Measure('ds', domain=mesh, subdomain_data=facets)
dV = Measure('dx', domain=mesh, subdomain_data=cells)

facets.set_all(0)
# a cyclic shear on top
shear = Expression(('0.0001*sin(2.*pi*f*time)','0.0'),degree=4,f=0,time=0) 
bc1 = DirichletBC(Space,shear,top)
bc2 = DirichletBC(Space,(0.0,0.0),bottom)
bc = [bc1,bc2]

def fractional(index,arg,t,h,power):
	m=index+1
	temp=as_tensor([[0.,0.],[0.,0.]])
	while m <= t/h:
		hist=int(t/h-m)
		tensor=Function(TensorSpace)
		arg.retrieve(tensor.vector(),hist)
		temp += as_tensor((-1.)**m*gamma(power+1.)\
		/gamma(power-m+1.)/factorial(m)*tensor[i,j]\
		/(h**power),[i,j])
		m += 1
	return temp


#Plotting stress v/s strain curves
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pylab
pylab.rc('text', usetex=True )
pylab.rc('font', family='serif', serif='cm', size=30 )
pylab.rc('legend', fontsize=30)
pylab.rc(('xtick.major','ytick.major'), pad=15)

def compute(freq,loops):
	if os.path.exists(pwd+'hist/'): shutil.rmtree(pwd+'hist/') 

	# a class in dolfin saves the data (binary) in time
	stress_history = TimeSeries(mesh.mpi_comm(), pwd+'hist/stressHist')
	strain_history = TimeSeries(mesh.mpi_comm(), pwd+'hist/strainHist')
	stress_history.parameters["clear_on_write"] = False
	strain_history.parameters["clear_on_write"] = False

	file_u = File (pwd+'displacements.pvd')

	t=t_start
	dt = 0.005/freq
	#Setting up the parameters in kg, s, N, m, Pa
	alpha = 0.844
	beta = 0.844-0.364
	tau0 = 214700.
	G0 = 2.152E2
	Ge = 1.185E4
	rho0 = 1200. #[kg/m3]
	
	#delta= Identity(Dim)
	epsilon= as_tensor(1.0/2.0*(u[i].dx(j)+u[j].dx(i)),(i,j))
	#epsilon0= as_tensor(1.0/2.0*(u0[i].dx(j)+u0[j].dx(i)),(i,j))

	sigma = as_tensor( 1.0/(1.0+ tau0**alpha/dt**alpha) * ( \
		- tau0**alpha*fractional(1,stress_history,t,dt,alpha)[i,j] \
		+ Ge*( epsilon[i,j] \
		+ tau0**alpha*fractional(0,strain_history,t,dt,alpha)[i,j] ) \
		+ G0*tau0**beta*fractional(0,strain_history,t,dt,beta)[i,j]) \
		, (i,j))
	f= Constant((0.0,0.0))

	Form = (rho0*(u[i]-2.*u0[i]+u00[i])/(dt*dt)*delu[i] \
		+ sigma[j,i]*delu[i].dx(j) \
		- rho0*f[i]*delu[i] )*dx
	Gain = derivative(Form,u,du)

	stresses = []
	strains = []
	u.interpolate(initialcond)
	u0.assign(u)
	u00.assign(u0)
	shear.f = freq
	tend=loops/freq

	while t<=tend:
		shear.time = t
		
		solve(Form == 0, u, bc, J=Gain, \
			solver_parameters={"newton_solver":{"linear_solver": "lu", "relative_tolerance": 1e-4} }, \
			form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}  )
		if t==0.25/freq : file_u<<(u,t)
		if t==0.75/freq : file_u<<(u,t)
		u00.assign(u0)
		u0.assign(u)
		actualStress = project(sigma,TensorSpace)
		stress_history.store(actualStress.vector(),t)		
		actualStrain = project(epsilon,TensorSpace)
		strain_history.store(actualStrain.vector(),t)
		P=(xlength/2.,ylength)
		sigma12 = actualStress(P)[1]
		epsilon12 = u(P)[0]/ylength
		stresses.append(sigma12) #in Pa
		strains.append(epsilon12*100.) #in %
		t = t + dt

	return stresses, strains


fig = pylab.figure(1, figsize=(12,8))
pylab.subplots_adjust(bottom=0.15)
pylab.subplots_adjust(left=0.20)
pylab.xlabel(r'strain $\varepsilon_{xy}$ in $\%$')
pylab.ylabel(r'stress $\sigma_{xy}$ in Pa')
pylab.grid(True)

freq = 0.2
stresses, strains = compute(freq,1)
#np.save('01Hz_stress.npy', stresses)
#np.save('01Hz_strain.npy', strains)
#stresses=np.load('01Hz_stress.npy')
#strains=np.load('01Hz_strain.npy')
pylab.plot(strains,stresses,color='red',marker='o',markersize=5,label='%(0).1f Hz'%{'0':freq})
pylab.savefig(pwd+'Hysteresis1.pdf')

elapsed = int(comp_time.time() - start_time)
e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
print('Frequency %.1f Hz is computed in %.0f h %.0f min %.0f s' % (freq, e_h, e_m, e_s )  )

freq=0.3
stresses, strains = compute(freq,1)
pylab.plot(strains,stresses,color='blue',marker='s',markersize=5,label='%(0).1f Hz'%{'0':freq})
pylab.legend(loc='best')
pylab.savefig(pwd+'Hysteresis1.pdf')

elapsed = int(comp_time.time() - start_time)
e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
print('Frequency %.1f Hz is computed in %.0f h %.0f min %.0f s' % (freq, e_h, e_m, e_s )  )

pylab.cla()
pylab.clf()
fig = pylab.figure(2, figsize=(12,8))
pylab.subplots_adjust(bottom=0.15)
pylab.subplots_adjust(left=0.20)
pylab.xlabel(r'strain $\varepsilon_{xy}$ in $\%$')
pylab.ylabel(r'stress $\sigma_{xy}$ in Pa')
pylab.grid(True)

freq = 0.2
stresses, strains = compute(freq,10)
pylab.plot(strains,stresses,'r-')
pylab.savefig(pwd+'Hysteresis2.pdf')
elapsed = int(comp_time.time() - start_time)
e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
print('Frequency %.1f Hz is computed in %.0f h %.0f min %.0f s' % (freq, e_h, e_m, e_s )  )


