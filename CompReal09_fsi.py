"""Computational reality 09, fluid structure interaction"""
__author__ = "B. Emek Abali"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

import time as comp_time
start_time = comp_time.time()

from dolfin import *
from ufl_legacy import indices
import numpy
set_log_level(50)

parameters["allow_extrapolation"]=True

xlength = 10.0	# in mm
ylength = 150.0	# in mm
mesh = RectangleMesh(Point(0.0, 0.0), Point(31*xlength, 1.5*ylength), 31*2,15*2)
Dim = mesh.geometric_dimension()

structure = CompiledSubDomain('x[0] >= 15.0*xl && x[0] <= 16.0*xl && x[1] <= yl',xl=xlength,yl=ylength)

interaction = CompiledSubDomain('(near(x[0],15.0*xl) && x[1] <= yl) || (near(x[0],16.0*xl) && x[1] <= yl) || (x[0] >= 15.0*xl && x[0] <= 16.0*xl && near(x[1], yl) )', xl=xlength, yl=ylength )

bottom = CompiledSubDomain('near(x[1],0.0)')
boundaries_all = CompiledSubDomain('on_boundary')
sub_domains = MeshFunction('size_t',mesh,Dim)
sub_domains.set_all(0)
structure.mark(sub_domains, 1)
mesh_f = create_meshview(sub_domains, 0)
mesh_s = create_meshview(sub_domains, 1)
#plot(mesh_f,interactive=True)
#plot(mesh_s,interactive=True)

#facets for solid: 0 for interior, 1 on bottom, 2 on interaction
facets_s = MeshFunction('size_t', mesh_s, Dim-1)
facets_s.set_all(0)
interaction.mark(facets_s,2)
bottom.mark(facets_s,1)
#facets for fluid: 0 for interior, 1 on boundary, 2 on interaction
facets_f = MeshFunction('size_t', mesh_f, Dim-1)
facets_f.set_all(0)
boundaries_all.mark(facets_f,1)
interaction.mark(facets_f,2)

cells_s = MeshFunction('size_t', mesh_s, Dim)
cells_f = MeshFunction('size_t', mesh_f, Dim)

dA = Measure('ds', domain=mesh_s, subdomain_data=facets_s)
dV = Measure('dx', domain=mesh_s, subdomain_data=cells_s)
da = Measure('ds', domain=mesh_f, subdomain_data=facets_f)
dv = Measure('dx', domain=mesh_f, subdomain_data=cells_f)

N = FacetNormal(mesh_s)
n = FacetNormal(mesh_f)

S_s_space = FunctionSpace(mesh_s, 'CG', 1)
V_s_space = VectorFunctionSpace(mesh_s, 'CG', 1)
T_s_space = TensorFunctionSpace(mesh_s, 'CG', 1)

S_f_space = FunctionSpace(mesh_f, 'CG', 1)
V_f_space = VectorFunctionSpace(mesh_f, 'CG', 1)
T_f_space = TensorFunctionSpace(mesh_f, 'CG', 1)

S_f_ele = FiniteElement("CG", mesh_f.ufl_cell(), 1)
V_f_ele = VectorElement("CG", mesh_f.ufl_cell(), 1)
SV_f_space = FunctionSpace( mesh_f, MixedElement([S_f_ele, V_f_ele]) )


t=0.0 # in s 
dt=0.01	# in s
t_end=5.0 # in s

pwd = 'results/CR09/'

file_u_s = File(pwd + 'u_s.pvd')
file_v_f = File(pwd + 'v_f.pvd')
file_p_f = File(pwd + 'p.pvd')

u_s_ = Function(V_s_space, name='u')
v_f_ = Function(V_f_space, name='v')
p_f_ = Function(S_f_space, name='p')

i,j,k,l,m = indices(5)
delta = Identity(Dim)
f= Constant((0.0,0.0))
f_param = {"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}

def s_solve(u0,u00,sigma_f,t):
	rho_s = 8.3E-9 #tonne/mm^3
	nu_s = 0.3
	E_s = 200E3 #in MPa
	mu_s = E_s/(2.*(1.+nu_s))
	lam_s = 2.*mu_s*nu_s/(1.-2.*nu_s)
	sigma_s = project(sigma_f,T_s_space, solver_type="mumps", form_compiler_parameters=f_param)
	#plot(facets_s, interactive=True)  
	displ_s_bottom = Expression(('A*sin(2.*pi*nu*t)','0.0'),degree=4, A=0.50, nu=10., t=t)
	bc_s = [DirichletBC(V_s_space, displ_s_bottom, facets_s, 1)]
	u_s = Function(V_s_space)
	del_u = TestFunction(V_s_space)
	du = TrialFunction(V_s_space)

	F_s = as_tensor( u_s[k].dx(i) + delta[k,i], (k,i) )
	J_s = det(F_s)
	C_s = as_tensor( F_s[k,i]*F_s[k,j], (i,j) )
	E_s = as_tensor( 1./2.*(C_s[i,j]-delta[i,j]), (i,j) )
	S_s = as_tensor( lam_s*E_s[k,k]*delta[i,j] + \
		2.*mu_s*E_s[i,j], (i,j) )
	P_s = as_tensor( F_s[i,j]*S_s[k,j], (k,i) )

	t_hat = as_tensor( J_s*inv(F_s)[k,j]*sigma_s[j,i]*N[k] , (i,) )

	Form_s = ( rho_s*(u_s-2.*u0_s+u00_s)[i]/(dt*dt)*del_u[i] \
		+ P_s[k,i]*del_u[i].dx(k) - rho_s*f[i]*del_u[i] )*dV \
		- t_hat[i]*del_u[i]*dA(2)
	Gain_s = derivative(Form_s, u_s, du)

	solve(Form_s == 0, u_s, bc_s, J=Gain_s, solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-3} }, form_compiler_parameters=f_param)

	v_s = project( (u_s-u0_s)/dt, V_s_space, solver_type="mumps", form_compiler_parameters=f_param)
	#plot(v_s, interactive=True)

	return u_s, v_s

def m_solve(v_s):
	# artificial viscosity
	a = 1.0E-11	# MPa/s 
	del_w = TestFunction(V_f_space)
	w = Function(V_f_space)
	dw = TrialFunction(V_f_space)
	bc_m = [DirichletBC(V_f_space, v_s, facets_f,2)]

	Form_m = a*sym(grad(w))[i,j]*del_w[i].dx(j)*dv
	Gain_m = derivative(Form_m, w, dw)

	solve(Form_m == 0, w, bc_m, J=Gain_m, solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-3} }, form_compiler_parameters=f_param)
			
	return w

def f_solve(w,v0_f):
	rho_f = 998.21E-12	# in tonne / mm^3
	mu_f = 1001.6E-12	# in N s / mm^2		
	lam_f = 1.0E-2		# in N s / mm^2
	bc1 = DirichletBC(SV_f_space.sub(0),Constant(0.1),facets_f,1)
	bc2 = DirichletBC(SV_f_space.sub(1),w,facets_f,2)		
	bc_f = [bc1,bc2]
	
	u_f = Function(SV_f_space)
	u0_f = Function(SV_f_space)
	del_u_f = TestFunction(SV_f_space)
	du_f = TrialFunction(SV_f_space)		
	
	p, v_f = split(u_f)
	del_p, del_v = split(del_u_f)

	tau_f = as_tensor(lam_f*v_f[k].dx(k)*delta[i,j] \
	+ mu_f*sym(grad(v_f))[i,j], (i,j))

	Form_f_p = (v_f[i].dx(i)*del_p + w[i].dx(i)*del_p \
	- (v_f-w)[i]*del_p.dx(i) )*dv + (v_f-w)[i]*del_p*n[i]*da(2)
	Form_f_v = (rho_f*(v_f-v0_f)[i]/dt*del_v[i] 
	+ rho_f*v_f[i].dx(j)*w[j]*del_v[i] + rho_f*v_f[i]*w[k].dx(k)*del_v[i] \
	+ p.dx(i)*del_v[i] - rho_f*v_f[i]*(v_f-w)[j]*del_v[i].dx(j) \
	+ tau_f[j,i]*del_v[i].dx(j) - rho_f*f[i]*del_v[i])*dv \
	+ (rho_f*v_f[i]*(v_f-w)[j] - tau_f[j,i])*del_v[i]*n[j]*da(1)
	
	Form_f = Form_f_p + Form_f_v
	Gain_f = derivative(Form_f, u_f, du_f)

	solve(Form_f == 0, u_f, bc_f, J=Gain_f, solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-3} }, form_compiler_parameters=f_param)

	p, v_f = u_f.split(deepcopy=True)
	sigma_f = project(-p*delta+tau_f, T_f_space, solver_type="mumps", form_compiler_parameters=f_param)

	return p, v_f, sigma_f
	

u0_s = Function(V_s_space)
u00_s = Function(V_s_space)
sigma_f = Function(T_f_space)
v0_f = Function(V_f_space)

while t<t_end:
	t += dt
	print("time: ", t)
	# inner loop for convergence between fluid and structure
	for ii in range(3):
		u_s, v_s = s_solve(u0_s,u00_s,sigma_f,t)
		w = m_solve(v_s)
		p, v_f, sigma_f = f_solve(w,v0_f)

	u00_s.assign(u0_s)
	u0_s.assign(u_s)
	v0_f.assign(v_f)
	for x in mesh_f.coordinates(): x[:] += dt*w(x)[:]
	u_s_.assign(u_s)
	file_u_s << (u_s_,t)
	v_f_.assign(v_f)
	file_v_f << (v_f_,t)
	p_f_.assign(p)
	file_p_f << (p_f_,t)
