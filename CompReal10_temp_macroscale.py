"""Computational reality 10, temperature distribution in a macroscopic body"""
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

rho=7860.0      # mass density of steel in kg/m^3
c=624.0	        # heat capacity in J/(kg K)
kappa=30.1	# thermal conductivity in W/(m K)
h=18.0		# heat convection out of the surface into ambient in W/(m^2 K)
Ta=300.0	# ambient temperature in K
l=0.1		# length in x and y directions in m 
thickness=0.001	# thickness of the plate in m
P=3.0e6		# laser power in W/kg 
speed=0.02	# laser speed in m/s
# Create mesh and define function space
mesh = BoxMesh(Point(0.0,0.0,0.0), Point(l,l,thickness), 200,200,2)
Space = FunctionSpace(mesh, 'P', 1)
cells = MeshFunction('size_t',mesh,mesh.topology().dim())
facets = MeshFunction('size_t',mesh,mesh.topology().dim()-1)
da = Measure('ds', domain=mesh, subdomain_data=facets)
dv = Measure('dx', domain=mesh, subdomain_data=cells)
t=0.0
t_end=50.0
Dt=0.1
initial_T = Expression("Tini", degree=1, Tini=Ta)
T0 = interpolate(initial_T, Space)
Laser = Expression("P*exp(-50000.0*(pow(x[0]-0.5*l*(1+0.5*sin(2*pi*t/t_e)), 2)+pow(x[1]-velo*t, 2)))",degree=4,P=P,t=0,t_e=t_end/10.,l=l,velo=speed)
T = TrialFunction(Space)
del_T = TestFunction(Space)
Form = (rho*c/Dt*(T-T0)*del_T \
	+ kappa*T.dx(i)*del_T.dx(i) \
	- rho*Laser*del_T ) * dv \
	+ h*(T-Ta)*del_T*da

left=lhs(Form)
right=rhs(Form)

A = assemble(left) 	# non-changing by time stepping
b = None		# dynamically assembled acc. to time
T = Function(Space)
file_T = File("results/CR10/T.pvd")
for t in numpy.arange(0,t_end,Dt):
	print("Time ",t)
	Laser.t=t
	b=assemble(right, tensor=b) 
	solve(A, T.vector(), b, 'cg')
	if t == int(t): file_T << (T,t)
	T0.assign(T)
