"""Computational reality 12, channel flow of Navier-Stokes-Fourier fluid"""
__author__ = "B. Emek Abali"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

import time as comp_time
start_time = comp_time.time()

from dolfin import *
from ufl_legacy import indices
import numpy
set_log_level(50)

xlength = 0.5 # m
ylength = 0.1 # m
mesh=RectangleMesh(Point(0.0,-ylength/2.0), Point(xlength,ylength/2.0), 200,40)
Dim = mesh.geometric_dimension()

TensorSpace = TensorFunctionSpace(mesh, 'CG', 1)
VectorSpace = VectorFunctionSpace(mesh, 'CG', 1)
ScalarSpace = FunctionSpace(mesh, 'CG', 1)
scalar_ele = FiniteElement("CG", mesh.ufl_cell(), 1)
vector_ele = VectorElement("CG", mesh.ufl_cell(), 1)
# p, v, T
Space = FunctionSpace( mesh, MixedElement([scalar_ele, vector_ele, scalar_ele]) )

cells = MeshFunction('size_t',mesh,Dim)
facets = MeshFunction('size_t',mesh,Dim-1)
da =  Measure('ds', domain=mesh, subdomain_data=facets)
dv = Measure('dx', domain=mesh, subdomain_data=cells)

left = CompiledSubDomain('near(x[0],0) && on_boundary')
right = CompiledSubDomain('near(x[0],l) && on_boundary',l=xlength)
bottom = CompiledSubDomain('near(x[1],-yl/2.0)',yl=ylength)
top = CompiledSubDomain('near(x[1],yl/2.0)',yl=ylength)

v_noslip = Constant((0.0, 0.0))
pL = Expression('100000.0+10*t',degree=1,t=0)
pR = Constant(100000.0)
Tini = 300. #K
bc1=DirichletBC(Space.sub(0), pL, left)
bc2=DirichletBC(Space.sub(0), pR, right)
bc3=DirichletBC(Space.sub(1), v_noslip, bottom)
bc4=DirichletBC(Space.sub(1), v_noslip, top)
bc5=DirichletBC(Space.sub(1).sub(1), 0.0, left)
bc6=DirichletBC(Space.sub(1).sub(1), 0.0, right)
bc7=DirichletBC(Space.sub(2), Tini, top)
bc8=DirichletBC(Space.sub(2), Tini, bottom)

bc=[bc1,bc2,bc3,bc4,bc5,bc6,bc7,bc8]
u_init = Expression(('p0','0.0','0.0','T0'), degree=1, p0=100000.0,T0=Tini)

i, j, k, l =indices(4)
n = FacetNormal(mesh)
t = 0.0
Dt = 50.
t_end = 500.

test = TestFunction(Space)
du = TrialFunction(Space)
u0 = Function(Space)
u = Function(Space)
u0 = interpolate(u_init,Space)
u = interpolate(u_init,Space)
p0, v0, T0 = split(u0)
p, v, T = split(u)
delp, delv, delT = split(test)
delta = Identity(Dim)

#water approx. at 300 K
rho = 995.7 #kg/m^3
mu = 0.8 #Ns/m^2 = kg /s /m
lambada = mu*1E5
c = 4180. #J/(kgK)
kappa = 0.58 #W/(m K)
h=18.0 #W/(m^2 K)
Tamb = Tini

d = sym(grad(v))
dsigma = as_tensor( lambada*d[k,k]*delta[i,j] + 2.0*mu*d[i,j], (i,j))
sigma = as_tensor( -p*delta[i,j] + dsigma[i,j], (i,j) )
q = as_tensor( -kappa*T.dx(i), (i,))
f = Constant((0,0))
r = Constant(0)

Form = (  v[i].dx(i)*delp \
	+ rho*(v-v0)[i]/Dt*delv[i] \
	+ rho*v[j]*v[i].dx(j)*delv[i] \
	+ p.dx(i)*delv[i] \
	+ dsigma[j,i]*delv[i].dx(j) \
	- rho*f[i]*delv[i]
	+ rho*c/T*(T-T0)/Dt*delT \
	+ rho*c/T*v[i]*T.dx(i)*delT \
	- q[i]/T*delT.dx(i) \
	- rho*r/T*delT \
	- 1./T*sigma[i,j]*v[j].dx(i)*delT \
	+ 1./T**2*q[i]*T.dx(i)*delT \
	)*dv 

Gain = derivative(Form, u, du)

pwd='results/CR12/'
file_p = File(pwd+'pressure.pvd')
file_v = File(pwd+'velocity.pvd')
file_T = File(pwd+'temperature.pvd')

for t in numpy.arange(0,t_end,Dt):
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('time: %.2e s computed in %.0f h %.0f min %.0f s' % (t, e_h, e_m, e_s )  )

	pL.t = t
	solve(Form == 0, u, bc, J=Gain, \
		solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-5} }, \
		form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}  )
	file_p << (u.split()[0], t)
	file_v << (u.split()[1], t)
	file_T << (u.split()[2], t)
	u0.assign(u)
