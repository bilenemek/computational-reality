"""Computational Reality 13, thermoviscoelasticity"""
__author__ = "B. Emek Abali"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

import time as comp_time
start_time = comp_time.time()

from dolfin import *
from ufl_legacy import indices
import numpy
set_log_level(50)

#units: mm, 1000 kg=tonne, s, MPa, mJ, K
delta = Identity(3)
f = Constant((0.0, 0.0, -9810.))
r = 0.
Tref = 293.15 #in K
Tamb=Tref

# Material data of P265GH (St 45.8) from VDI Waermeatlas, at 293.15 K
rho0 = 7850.0E-9 #in kg / mm^3
kappa = 57.0 #in mJ / (s mm K)
capacity = 430.0E6 #in mJ / (ton K)
alpha = 12.2E-6 #in 1/K at 373.15 K
EModul = 211.E+3 #in MPa
nu = 0.28
h = 10.E-3 #in mJ / (s m^2 K)
mu1 = 1.E+6 #in MPa / s
mu2 = 3.E+6 #in MPa / s

tMax = 5.0
Dt = 0.5
t = 0.0

xMin, xMax, xElements = 0.0, 100.0, 10
yMin, yMax, yElements = -10., +10., 10
zMin, zMax, zElements = +10., -10., 10
mesh = BoxMesh(Point(xMin,yMin,zMin), Point(xMax,yMax,zMax), xElements,yElements,zElements)
Dim = mesh.geometric_dimension()
N = FacetNormal(mesh)
length = abs(xMax-xMin)

scalar_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
vector_ele = VectorElement("CG", mesh.ufl_cell(), 1)
# T, u
Space = FunctionSpace( mesh, MixedElement([scalar_ele, vector_ele]) )

cells = MeshFunction('size_t',mesh,Dim)
facets = MeshFunction('size_t',mesh,Dim-1)
dA = Measure('ds', domain=mesh, subdomain_data=facets)
dV = Measure('dx', domain=mesh, subdomain_data=cells)

left = CompiledSubDomain('near(x[0],l) && on_boundary',l=xMin)
right = CompiledSubDomain('near(x[0],l) && on_boundary',l=xMax)
back = CompiledSubDomain('near(x[1],l) && on_boundary',l=yMin)
front = CompiledSubDomain('near(x[1],l) && on_boundary',l=yMax)
bottom = CompiledSubDomain('near(x[2],l) && on_boundary',l=zMin)
top = CompiledSubDomain('near(x[2],l) && on_boundary',l=zMax)

facets.set_all(0)
right.mark(facets, 1)
tHat = Expression(('A*t','0.','0.'), degree=1, A=250., t=0.)
bc = [DirichletBC(Space.sub(1), Constant((0.0, 0.0, 0.0)), left),\
DirichletBC(Space.sub(1).sub(1), Constant(0.0), right),\
DirichletBC(Space.sub(1).sub(2), Constant(0.0), right)]

dunkn = TrialFunction(Space)
test = TestFunction(Space)
delT, delu = split(test)

unkn = Function(Space)
unkn0 = Function(Space)
unkn00 = Function(Space)

unkn_init = Expression(('T_ini','0','0','0'),degree=1,T_ini=Tref)
unkn = interpolate(unkn_init,Space)
unkn0.assign(unkn)
unkn00.assign(unkn0)

T, u = split(unkn)
T0, u0 = split(unkn0)
T00, u00 = split(unkn0)

i, j, k, l = indices(4) 
delta = Identity(Dim)
F = as_tensor(u[i].dx(j)+delta[i,j], (i,j))
F0 = as_tensor(u0[i].dx(j)+delta[i,j], (i,j))
C = as_tensor(F[k,i]*F[k,j], (i,j))
C0 = as_tensor(F0[k,i]*F0[k,j], (i,j))
E = as_tensor(1./2.*(C[i,j]-delta[i,j]), (i,j))
E0 = as_tensor(1./2.*(C0[i,j]-delta[i,j]), (i,j))
lambada = EModul * nu / (1. + nu) / (1. - 2. * nu)
mu = 0.5 * EModul / (1. + nu)
C_ = as_tensor(lambada*delta[i,j]*delta[k,l]+mu*delta[i,k]*delta[j,l]+mu*delta[i,l]*delta[j,k], (i,j,k,l))
alp = as_tensor(alpha*delta[i,j], (i,j))
eStress = as_tensor(-C_[i,j,k,l]*alp[k,l]*(T-Tref) + C_[i,j,k,l]*E[k,l], (i,j))
dStress = as_tensor((mu1-mu2)/3.*(E-E0)[k,k]/Dt*delta[i,j] + mu2*(E-E0)[i,j]/Dt, (i,j))
S = as_tensor(eStress[i,j]+dStress[i,j], (i,j))
P = as_tensor(F[j,l]*S[i,l], (i,j))
eta = as_tensor( capacity*ln(T/Tref) + C_[i,j,k,l]*alp[k,l]*1./rho0*E[i,j] , ())
eta0 = as_tensor( capacity*ln(T0/Tref) + C_[i,j,k,l]*alp[k,l]*1./rho0*E0[i,j] , ())
Q = as_tensor(-kappa*T.dx(i), (i,))

Form = (rho0*(u-2.*u0+u00)[i]/Dt/Dt*delu[i] \
	+ P[j,i]*delu[i].dx(j) - rho0*f[i]*delu[i]  \
	+ rho0/T*(eta-eta0)*delT - Dt/T*Q[i]*delT.dx(i) - Dt*rho0*r/T*delT \
	- 1./T*dStress[i,j]*(E-E0)[i,j]*delT + Dt/T**2*Q[i]*T.dx(i)*delT )*dV \
	+ Dt/T*h*(T-Tamb)*delT*(dA(0)+dA(1)) - tHat[i]*delu[i]*dA(1)

Gain = derivative(Form, unkn, dunkn)

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pylab
pylab.rc('text', usetex=True )
pylab.rc('font', family='serif', serif='cm', size=20 )
pylab.rc('legend', fontsize=20)
pylab.rc(('xtick.major','ytick.major'), pad=5)
pylab.subplots_adjust(top=0.90)
pylab.subplots_adjust(bottom=0.2)
pylab.subplots_adjust(left=0.20)
pylab.subplots_adjust(right=0.8)

fig = pylab.figure(1, figsize=(14,10))
ax1 = fig.add_subplot(111)
ax1.grid(True, axis='x')
ax1.set_xlabel('$u_1/l_0$ in $\%$')
ax1.set_ylabel('$F/A$ in MPa', color='r')
ax1.tick_params(axis='y', colors='r')
ax1.grid(True, axis='y', color='r')
ax2 = ax1.twinx()
ax2.set_ylabel('$(T-T_\mathrm{ref})$ in K', color='b')
ax2.tick_params(axis='y', colors='b')
ax2.grid(True, axis='y', color='b')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))

pwd='results/CR13/'
file_u = File(pwd+'displ.pvd')
file_T = File(pwd+'temp.pvd')
strain,stress,temp =[],[],[]

while t < tMax:
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('time: %.2e s computed in %.0f h %.0f min %.0f s' % (t, e_h, e_m, e_s )  )
	tHat.t = t
	solve(Form == 0, unkn, bc, J=Gain, \
		solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-5} }, \
		form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}  )

	file_T << (unkn.split()[0], t)
	file_u << (unkn.split()[1], t)

	strain.append(unkn.split()[1](xMax,0.,0.)[0]/length*100.)
	stress.append(tHat(xMax,0.,0.)[0])
	temp.append(unkn.split()[0](xMax/2.,0.,0.)-Tref)
	ax1.plot(strain,stress, 'o-', color='r')
	ticks = numpy.linspace(numpy.array(strain).min(), numpy.array(strain).max(), 4)
	ax1.set_xticks(ticks)
	ax1.set_xticklabels(['%1.2f' % i_ticks for i_ticks in ticks] )
	ax2.plot(strain,temp, 'd-', color='b')
	fig.savefig(pwd+'CompReal13_tensiletest.pdf')
	unkn00.assign(unkn0)
	unkn0.assign(unkn)
	t += Dt

