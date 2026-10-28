"""Computational reality 11, temperature distribution in a microscopic body"""
__author__ = "B. Emek Abali"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

import time as comp_time
start_time = comp_time.time()

from dolfin import *
from ufl import indices
i, j =indices(2)
import numpy

parameters["allow_extrapolation"]=True
parameters["form_compiler"]["cpp_optimize"] = True
set_log_level(50)

rho=19.3E-6     # mass density of gold (Au) in mug/mum^3]
c=129.0E6	# heat capacity in fJ/(mug K)
kappa=318.0E-3	# thermal conductivity in fJ/(mum K ps)
tau_T = 89.286	# in ps
tau_q = 0.7438	# in ps
h=18.0E-9	# natural convection coefficient in fJ/(ps mum^2 K)]
emis=0.47	# emissivity of gold not polished
sigma=5.670E-17	# Stefan-Boltzmann constant in fJ/(ps mum**2)
Ta=300.0	# ambient temperature in K
l=100.0		# length in mum 
th=5.0		# thickness in mum
P=30.0E3	# laser power in fJ/(mum ps)

mesh = BoxMesh(Point(0.0,0.0,0.0), Point(l,th,th), 200,10,10)
Space = FunctionSpace(mesh, 'P', 1)
VectorSpace = VectorFunctionSpace(mesh, 'P', 1)
facets = MeshFunction('size_t',mesh,mesh.topology().dim()-1)
cells = MeshFunction('size_t',mesh,mesh.topology().dim())
da = Measure('ds', domain=mesh, subdomain_data=facets)
dv = Measure('dx', domain=mesh, subdomain_data=cells)
t=0.0
t_end=100000.0 #0.1 mus
Dt=1000.0
initial_T = Expression("Tini", degree=1, Tini=Ta)
T0 = interpolate(initial_T, Space)
Laser = Expression("P*exp(-1.0*(pow(x[0]-2, 2)+pow(x[1], 2)+pow(x[2]-2.5, 2)))",degree=4,P=P)
T = Function(Space)
del_T = TestFunction(Space)
dT = TrialFunction(Space)
q0 = Function(VectorSpace)
G = as_tensor(T.dx(i), (i,))
G0 = as_tensor(T0.dx(i), (i,))
q = as_tensor(Dt/(Dt+tau_q)*(tau_q/Dt*q0[i]-kappa*(1+tau_T/Dt)*G[i]+kappa*tau_T/Dt*G0[i]), (i,))
Form = (rho*c/Dt*(T-T0)*del_T \
	- q[i]*del_T.dx(i) \
	- rho*Laser*del_T ) * dv \
	+ (h*(T-Ta) \
	+  emis*sigma*(T**4-Ta**4) )*del_T*da
Gain = derivative(Form, T, dT)

file_T = File("results/CR11/T.pvd")
for t in numpy.arange(0,t_end,Dt):
	print("Time ",t)
	if t>=2000.: Laser.P=0
	solve(Form == 0, T, [], J=Gain, \
		solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-3} }, \
		form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}  )

	if t == int(t): 
		file_T << (T,t)
		elapsed = int(comp_time.time() - start_time)
		e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
		print('Time %.4f s computed in %.0f h %.0f min %.0f s' % (t, e_h, e_m, e_s )  )
	q0 = project(q,VectorSpace)
	T0.assign(T)
