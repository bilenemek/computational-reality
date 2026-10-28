"""Computational reality 15, conducting wire"""
__author__ = "B. Emek Abali"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

import time as comp_time
start_time = comp_time.time()

from dolfin import *
from ufl import indices
import numpy
set_log_level(50)
#units: m, kg, s, A, V, K
delta = Identity(3)
Tref = 300. #in K
t_alpha = 3.9E-3 #in 1/K
varsigma0 = 5.8E7 #in S/m or in 1/(Ohm m)
Tamb=Tref
rho = 8960. #in kg / m^3
kappa = 400.0 #in W/(m K)
capacity = 390. #in J / (kg K)
#approximate values: for convection over boundaries
#free convection in air h=10
#forced convection in air h=100
#forced convection in water h=1000
h = 10. #in W / (m^2 K)

tMax = 1500.0
Dt = 50.0
t = 0.0

xMin, xMax, xElements = 0.0, 1.0, 200
yMin, yMax, yElements = -0.005, +0.005, 2
zMin, zMax, zElements = -0.005, +0.005, 2
mesh = BoxMesh(Point(xMin,yMin,zMin), Point(xMax,yMax,zMax), xElements,yElements,zElements)
N = FacetNormal(mesh)

Scalar = FunctionSpace(mesh, 'CG', 1)
Vector = VectorFunctionSpace(mesh, 'CG', 1)
Tensor = TensorFunctionSpace(mesh, 'CG', 1)
# phi, T
scalar = FiniteElement("CG", mesh.ufl_cell(), 1) 
Space = FunctionSpace( mesh, MixedElement([scalar, scalar]) )

cells = MeshFunction('size_t', mesh, mesh.topology().dim())
facets = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
da = Measure('ds', domain=mesh, subdomain_data=facets)
dv = Measure('dx', domain=mesh, subdomain_data=cells)

left = CompiledSubDomain('near(x[0],l) && on_boundary',l=xMin)
right = CompiledSubDomain('near(x[0],l) && on_boundary',l=xMax)
facets.set_all(0)
bc = [DirichletBC(Space.sub(0), 0.0, left),\
DirichletBC(Space.sub(0), 0.1, right),]

dunkn = TrialFunction(Space)
test = TestFunction(Space)
del_phi, del_T = split(test)

unkn = Function(Space)
unkn0 = Function(Space)

unkn_init = Expression(('0.0','T_ini'),degree=1,T_ini=Tref)
unkn0 = interpolate(unkn_init,Space)
unkn = interpolate(unkn_init,Space)

phi, T = split(unkn)
phi0, T0 = split(unkn0)

i, j, k, l = indices(4) 
delta = Identity(3)
varsigma = varsigma0/(1.0 + t_alpha*(T-Tref))
J = -varsigma*grad(phi)
q = -kappa*grad(T)
r = Constant(0.0)

Form = (-J[i]*del_phi.dx(i) + rho*capacity*(T-T0)/Dt*del_T/T - q[i]*(del_T/T).dx(i) - rho*r*del_T/T + J[i]*phi.dx(i)*del_T/T )*dv + h*(T-Tamb)*del_T/T*da
Gain = derivative(Form, unkn, dunkn)

pwd='results/CR15/'
file_phi = File(pwd+'phi.pvd')
file_T = File(pwd+'temp.pvd')

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pylab
pylab.rc('text', usetex=True )
pylab.rc('font', family='serif', serif='cm', size=20 )
pylab.rc('legend', fontsize=20)
pylab.rc(('xtick.major','ytick.major'), pad=10)
pylab.subplots_adjust(top=0.90)
pylab.subplots_adjust(bottom=0.17)
pylab.subplots_adjust(left=0.20)
pylab.subplots_adjust(right=0.95)
fig = pylab.figure(1, figsize=(14,10))
ax1 = fig.add_subplot(111)
ax1.set_xlabel('$t$ in s')
ax1.set_ylabel('$T$ in K')
ax1.set_xlim([0.0,tMax])
t_plot, T_plot = [0], [Tref]

while t < tMax:
	print ('time: ',t,' T ',T_plot[-1])
	solve(Form == 0, unkn, bc, J=Gain, \
		solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-5} }, \
		form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}  )

	file_T << (unkn.split()[1], t)
	file_phi << (unkn.split()[0], t)

	t_plot.append(t)
	T_plot.append(unkn.split()[1](xMax/2.,0.,0.))
	ax1.plot(t_plot,T_plot, 'o-', color='r')
	fig.savefig(pwd+'CompReal15_temp.pdf')
	unkn0.assign(unkn)
	t += Dt

