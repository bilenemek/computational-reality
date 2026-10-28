"""Computational reality 07, linear fluid flow"""
__author__ = "B. Emek Abali"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

import time as comp_time
start_time = comp_time.time()

from dolfin import *
from ufl import indices
import numpy
set_log_level(50)
xlength = 5.0
ylength = 1.0
mesh=RectangleMesh(Point(0.0,-ylength/2.0), Point(xlength,ylength/2.0),100,20)
V = VectorElement("CG", mesh.ufl_cell(), 1) #for velocity
P = FiniteElement("CG", mesh.ufl_cell(), 1) #for pressure
Space = FunctionSpace( mesh, MixedElement([P, V]) )

left = CompiledSubDomain('near(x[0],0) && on_boundary')
right = CompiledSubDomain('near(x[0],l) && on_boundary',l=xlength)
bottom = CompiledSubDomain('near(x[1],l) && on_boundary',l=-ylength/2.0)
top = CompiledSubDomain('near(x[1],l) && on_boundary',l=+ylength/2.0)

facets = MeshFunction('size_t',mesh,mesh.topology().dim())
cells = MeshFunction('size_t',mesh,mesh.topology().dim()-1)
da = Measure('ds', domain=mesh, subdomain_data=facets)
dv = Measure('dx', domain=mesh, subdomain_data=cells)
n = FacetNormal(mesh)

facets.set_all(0)
left.mark(facets, 1)
right.mark(facets, 2)

#setting the Dirichlet conditions
v_noslip = Constant(('0.0', '0.0'))
p_in = Expression('100.0+10.0*t', degree=2, t=0)
p_out = Constant('100.0')
tL = -p_in*n
tR = -p_out*n
p_bc1=DirichletBC(Space.sub(0), p_in, left)
p_bc2=DirichletBC(Space.sub(0), p_out, right)
v_bc1=DirichletBC(Space.sub(1), v_noslip, bottom)
v_bc2=DirichletBC(Space.sub(1), v_noslip, top)
v_bc3=DirichletBC(Space.sub(1).sub(1), 0.0, left)
v_bc4=DirichletBC(Space.sub(1).sub(1), 0.0, right)

bc=[p_bc1,p_bc2,v_bc1,v_bc2,v_bc3,v_bc4]
f = Expression(('0.0', '0.0'), degree=2)
# material constants
a = 0.1
Re = 100.
i, j, k, l = indices(4)
t = 0.0
t_bc = 0.5
t_end = 10.0
dt = 0.05

test = TestFunction(Space)
du = TrialFunction(Space)
u0 = Function(Space) # solution from the last time slice
u = Function(Space) # current solution
u_init = Expression(('p0','0.0','0.0'), degree=2, p0=100.0)
u0.interpolate(u_init)
u.interpolate(u_init) 
# split() sets pointer to the memory address, 
# so it is a direct access, no deep copy
p0, v0 = split(u0)
p, v = split(u)
del_p, del_v = split(test)
Form = (v[i].dx(i)*del_p + (v-v0)[i]/dt*del_v[i] \
	+ v[j]*v[i].dx(j)*del_v[i] + p.dx(i)*del_v[i] \
	+ 1.0/(a*Re)*v[k].dx(k)*del_v[i].dx(i) \
	+ 1.0/Re*sym(grad(v))[j,i]*del_v[i].dx(j) \
	- f[i]*del_v[i] ) *dv

Gain = derivative(Form, u, du)
file_p = File('results/CR07/pres.pvd')
file_v = File('results/CR07/velo.pvd')
# time loop
while t<= t_end:
	t = t + dt
	if t<t_bc: p_in.t = t
	else: p_in.t = t_bc

	solve(Form == 0, u, bc, J=Gain, \
		solver_parameters={"newton_solver":{"linear_solver": "lu", "relative_tolerance": 1e-3} }, \
		form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}  )

	file_p << (u.split()[0],t)
	file_v << (u.split()[1],t)
	u0.assign(u)

	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('time: %.4f s computed in %.0f h %.0f min %.0f s' % (t, e_h, e_m, e_s )  )

