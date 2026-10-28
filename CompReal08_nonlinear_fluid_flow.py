"""Computational reality 08, nonlinear fluid flow"""
__author__ = "B. Emek Abali"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

import time as comp_time
start_time = comp_time.time()

from dolfin import *
from ufl import indices
set_log_level(50)

xlength = 5.0 # m
ylength = 1.0 # m
mesh_=RectangleMesh(Point(0.0,-ylength/2.0), Point(xlength,ylength/2.0), 100,20)
class Cut(SubDomain):
	def inside(self, x, on_boundary):
		return x[0]<xlength/2.0 and (x[1]>+ylength/4.0 or x[1]<-ylength/5.0)

domain = MeshFunction('size_t',mesh_,mesh_.topology().dim())
domain.set_all(0)
to_be_cut = Cut()
to_be_cut.mark(domain, 1)
mesh = SubMesh(mesh_, domain, 0)

V = VectorElement("CG", mesh.ufl_cell(), 2) #for velocity
P = FiniteElement("CG", mesh.ufl_cell(), 1) #for pressure
Space = FunctionSpace( mesh, MixedElement([P, V]) )

left = CompiledSubDomain('near(x[0],0) && on_boundary')
right = CompiledSubDomain('near(x[0],l) && on_boundary',l=xlength)
bottom1 = CompiledSubDomain('x[0]>xl/2.0 && near(x[1],-yl/2.0)',xl=xlength,yl=ylength)
bottom2 = CompiledSubDomain('x[0]<xl/2.0 && x[1]<-yl/5.0',xl=xlength,yl=ylength)
top1 = CompiledSubDomain('x[0]>xl/2.0 && near(x[1],yl/2.0)',xl=xlength,yl=ylength)
top2 = CompiledSubDomain('x[0]<xl/2.0 && x[1]>yl/4.0',xl=xlength,yl=ylength)
opening1 = CompiledSubDomain('near(x[0],xl/2.0) && x[1]<-yl/5.0',xl=xlength,yl=ylength)
opening2 = CompiledSubDomain('near(x[0],xl/2.0) && x[1]>yl/4.0',xl=xlength,yl=ylength)

facets = MeshFunction('size_t',mesh,mesh.topology().dim()-1)
cells = MeshFunction('size_t',mesh,mesh.topology().dim())
da_ = Measure('ds', domain=mesh, subdomain_data=facets)
dv_ = Measure('dx', domain=mesh, subdomain_data=cells)

v_noslip = Constant((0.0, 0.0))
pL = Expression('100000.0+1000.0*t', degree=2, t=0)
pR = Constant(100000.0)
p_bc1=DirichletBC(Space.sub(0), pL, left)
p_bc2=DirichletBC(Space.sub(0), pR, right)
p_bc3=DirichletBC(Space.sub(0), pR, opening1)
p_bc4=DirichletBC(Space.sub(0), pR, opening2)
v_bc1=DirichletBC(Space.sub(1), v_noslip, bottom1)
v_bc2=DirichletBC(Space.sub(1), v_noslip, bottom2)
v_bc3=DirichletBC(Space.sub(1), v_noslip, top1)
v_bc4=DirichletBC(Space.sub(1), v_noslip, top2)
v_bc5=DirichletBC(Space.sub(1).sub(1), 0.0, left)
v_bc6=DirichletBC(Space.sub(1).sub(1), 0.0, right)

bc=[p_bc1,p_bc2,p_bc3,p_bc4,v_bc1,v_bc2,v_bc3,v_bc4,v_bc5,v_bc6]
u_init = Expression(('p0','0.0','0.0'), degree=2, p0=100000.0)

i, j, k, l, m, n =indices(6)  
t = 0.0
t_end = 10.0
dt = 0.1

test = TestFunction(Space)
du = TrialFunction(Space)
u0 = Function(Space)
u = Function(Space)
u = interpolate(u_init,Space)
u0.assign(u)
p0, v0 = split(u0)
p, v = split(u)
delp, delv = split(test)
dp, dv = split(du)
delta = Identity(2)

# approximate values of HDPE melt, rho [kg/m^3], mu [Pa s], k [Pa]
rho, mu, lambada, k_fluid, B_fluid = 1450.0, 400000.0, 1E6, 250.0, 0.00001

# if II=0 (which is physically possible) Form will be singular, 
# to avoid that, add a small enough number to approximate an 
# accurate solution
II = as_tensor(1.0/2.0*sym(grad(v))[m,n]*sym(grad(v))[m,n]+0.000001, ()) 
I = as_tensor(sym(grad(v))[k,k], ())
#scalar is a tensor of rank zero


Form = (  v[i].dx(i)*delp \
	+ rho/dt*(v-v0)[i]*delv[i] \
	+ rho*v[j]*v[i].dx(j)*delv[i] \
	+ p.dx(i)*delv[i] \
	+ lambada*I*delv[i].dx(i) \
	+ (2.0*mu + 2.0*k_fluid/pi/sqrt(II) \
	*atan(sqrt(II)/B_fluid))*sym(grad(v))[j,i]\
	*delv[i].dx(j) \
	)*dv_

S=2.0*II
Gain = (  dv[i].dx(i)*delp \
	+ rho/dt*dv[i]*delv[i] \
	+ dv[j]*rho*v[i].dx(j)*delv[i] \
	+ v[j]*rho*dv[i].dx(j)*delv[i] \
	+ dp.dx(i)*delv[i] \
	+ lambada*dv[k].dx(k)*delv[i].dx(i) \
	+ 2.0*mu*sym(grad(dv))[k,j]*delv[j].dx(k) \
	- 2.0*k_fluid*2.0**0.5/(pi*S**(3.0/2.0))*sym(grad(v))[i,l] \
	*sym(grad(dv))[i,l]*atan(S**0.5/(2.0**0.5*B_fluid))\
	*sym(grad(v))[k,j]*delv[j].dx(k) \
	+ 4.0*B_fluid*k_fluid*sym(grad(v))[m,n]*sym(grad(dv))[m,n] \
	/(pi*S*(2.0*B_fluid**2+S))*sym(grad(v))[k,j]*delv[j].dx(k) \
	+ 2.0*k_fluid*2.0**0.5/(pi*S**0.5)*atan(S**0.5/(2.0**0.5*B_fluid)) \
	*sym(grad(dv))[k,j]*delv[j].dx(k) )*dv_

#Gain = derivative(Form, u, du)

pwd='results/CR08/'
file_p = File(pwd+'pressure.pvd')
file_v = File(pwd+'velocity.pvd')

# ---------------------------------------------------------time loop

while t<t_end:
	t += dt
	pL.t = t
	solve(Form == 0, u, bc, J=Gain, \
		solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-3} }, \
		form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}  )
	file_p << (u.split()[0], t)
	file_v << (u.split()[1], t)
	II_ = project( II, FunctionSpace(mesh,'P',1), solver_type="mumps",\
		form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2})
	sigma = as_tensor(-p*delta[j,i]+(2.0*mu+2.0*k_fluid/pi/sqrt(II_) \
	*atan(sqrt(II_)/B_fluid))*sym(grad(v))[j,i],(j,i))
	sigma_=project(sigma,TensorFunctionSpace(mesh,'P',1), solver_type="mumps",\
		form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2})

	u0.assign(u)
	
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('time: %.4f s computed in %.0f h %.0f min %.0f s, sigma12 = %.2f Pa' % (t, e_h, e_m, e_s, sigma_(xlength/2.0,ylength/4.0)[1] )  )

