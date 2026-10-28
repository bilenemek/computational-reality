"""Computational reality 17, thermoelectric coupling"""
__author__ = "B. Emek Abali"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html


import time as comp_time
start_time = comp_time.time()

from dolfin import *
from ufl_legacy import indices
import numpy
set_log_level(50)


class material_coefficient(UserExpression):
	def __init__(self, target_mesh, use_list, with_coeffs, **kwargs):
		super().__init__(**kwargs)
		self._mesh = target_mesh
		self.cell_list = use_list
		self.coeffs = with_coeffs
		self._element = kwargs["element"]
	def eval_cell(self, values, x, cell):
		values[0]= self.coeffs[ self.cell_list[cell.index] -1 ]

#units: m, kg, s, A, V, K
delta = Identity(3)
levicivita2 = as_matrix([ (0,1,-1) , (-1,0,1) , (1,-1,0) ])
levicivita3 = as_tensor([ ( (0,0,0),(0,0,1),(0,-1,0) ) , ( (0,0,-1),(0,0,0),(1,0,0) ) , ( (0,1,0),(-1,0,0),(0,0,0) ) ])
epsilon = levicivita3

#thermocouple of type E is made of 
#chromel (nickel-chromium alloy) and 
#is non-magnetic in reality, we also
#assume that it is non-polarizable
resistivity = 0.706E-6 #in Ohm m
varsigma = 1./resistivity #0.625 #in S/m
pi = 68E-6 #V/K
kappa = 19.0 #in W/(m K)
capacity = 390. #in J / (kg K)
alpha = 12.8E-6 #in 1/K
EModul = 186E+9 #in Pa
nu = 0.32
h = 10. #in J / (s m^2 K)

eps_0 = 8.85E-12 #in A s/(V m)
mu_0 = 12.6E-7 #in V s/(A m)

rho0 = 8500. #in kg / m^3
T_ref = 300.0 # K

tMax = 30.0
Dt = 1.0
t = 0.0

mesh = BoxMesh(Point(-0.05,-0.01,-0.01), Point(0.05,0.01,0.01), 50,10,10)
Dim = mesh.geometric_dimension()
N = FacetNormal(mesh)

Scalar = FunctionSpace(mesh, 'CG', 1)
Vector = VectorFunctionSpace(mesh, 'CG', 1)
Tensor = TensorFunctionSpace(mesh, 'CG', 1)
#phi, A, u, T
scalar = FiniteElement("CG", mesh.ufl_cell(), 1) 
vector = VectorElement("CG", mesh.ufl_cell(), 1) 
Space = FunctionSpace( mesh, MixedElement([scalar, vector, vector, scalar]) )

cells = MeshFunction('size_t', mesh, Dim)
facets = MeshFunction('size_t', mesh, Dim-1)
dV = Measure('dx', domain=mesh, subdomain_data=cells)
dA = Measure('ds', domain=mesh, subdomain_data=facets)

cells.set_all(0)
facets.set_all(0)
left = CompiledSubDomain('near(x[0],-0.05) && on_boundary')
right = CompiledSubDomain('near(x[0],0.05) && on_boundary')
boundaries = CompiledSubDomain('on_boundary')

#phi, A, u, T
bc_T = Expression('T_r + 1.0*time', degree=2, T_r=T_ref,time=0.)
bc01=DirichletBC(Space.sub(0), 0.0, right)
bc02=DirichletBC(Space.sub(2), Constant((0.,0.,0.)), right)
bc03=DirichletBC(Space.sub(3), T_ref, right)
bc04=DirichletBC(Space.sub(3), bc_T, left)

bc = [bc01,bc02,bc03,bc04]

dunkn = TrialFunction(Space)
test = TestFunction(Space)
unkn = Function(Space)
unkn0 = Function(Space)
unkn00 = Function(Space)

unkn_init = Expression(('0.','0.','0.','0.','0.','0.','0.','T_r'),degree=1,T_r=T_ref)
unkn00 = interpolate(unkn_init,Space)
unkn0.assign(unkn00)
unkn.assign(unkn0)

del_phi,del_A,del_u,del_T = split(test)
phi,A,u,T = split(unkn)
phi0,A0,u0,T0 = split(unkn0)
phi00,A00,u00,T00 = split(unkn00)

i, j, k, l = indices(4) 
delta = Identity(3)

lam = EModul * nu / (1.+nu) / (1.-2.*nu)
mu = 0.5 * EModul / (1.+nu)
C = as_tensor(lam*delta[i,j]*delta[k,l] + mu*delta[i,k]*delta[j,l] + mu*delta[i,l]*delta[j,k], (i,j,k,l))
alfa = alpha*delta

eps = sym(grad(u))
eps0 = sym(grad(u0))
v = as_tensor( (u-u0)[i]/Dt, (i,))
E = as_tensor(-phi.dx(i)-(A-A0)[i]/Dt, (i,))
E0 = as_tensor(-phi0.dx(i)-(A0-A00)[i]/Dt, (i,))
B = as_tensor(epsilon[i,j,k]*A[k].dx(j), (i,))
EE = as_tensor(E[i]+epsilon[i,j,k]*v[j]*B[k], (i,))

D = eps_0*E
D0 = eps_0*E0
H = 1./mu_0*B
JJ = as_tensor( varsigma*pi*T.dx(i) + varsigma*EE[i], (i,))
J = as_tensor( JJ[i] + D[j].dx(j)*v[i], (i,))
sigma = as_tensor( C[i,j,k,l]*(eps[k,l]-alfa[k,l]*(T-T_ref)), (i,j))
eta = as_tensor( capacity*ln(T/T_ref) + C[i,j,k,l]*alfa[k,l]/rho0*eps[i,j], ())
eta0 = as_tensor( capacity*ln(T0/T_ref) + C[i,j,k,l]*alfa[k,l]/rho0*eps0[i,j], ())
q = as_tensor(-kappa*T.dx(i)+varsigma*pi*T*EE[i], (i,))
FF = as_tensor(D[j].dx(j)*E[i] + epsilon[i,j,k]*J[j]*B[k], (i,))
f = Constant((0.,0.,0.))
r = Constant(0.0)

F_phi = (-(D-D0)[i]*del_phi.dx(i) - Dt*J[i]*del_phi.dx(i) )*dV + Dt*J[i]*del_phi*N[i]*dA
F_A = (eps_0*(A-2.*A0+A00)[i]/Dt/Dt*del_A[i] + 1./mu_0*A[i].dx(j)*del_A[i].dx(j) - J[i]*del_A[i] )*dV 
F_u = (rho0*(u-2.*u0+u00)[i]/Dt/Dt*del_u[i] + sigma[j,i]*del_u[i].dx(j) - rho0*f[i]*del_u[i] - FF[i]*del_u[i])*dV
F_T = (rho0*(eta-eta0)*del_T - Dt*q[i]*(del_T/T).dx(i) - Dt/T*rho0*r*del_T - Dt/T*JJ[i]*EE[i]*del_T )*dV + Dt/T*h*(T-T_ref)*del_T*dA

Form = F_phi + F_A + F_u + F_T
Gain = derivative(Form, unkn, dunkn)

pwd='results/CR17/'
file_phi = File(pwd+'phi.pvd')
file_A = File(pwd+'A.pvd')
file_E = File(pwd+'E.pvd')
file_B = File(pwd+'B.pvd')
file_u = File(pwd+'u.pvd')
file_T = File(pwd+'T.pvd')

phi_ = Function(Scalar, name='$\phi$')
A_ = Function(Vector, name='$A_i$')
u_ = Function(Vector, name='$u_i$')
T_ = Function(Scalar, name='$T$')
E_ = Function(Vector, name='$E_i$')
B_ = Function(Vector, name='$B_i$')

while t < tMax:

	if t <= 10.: bc_T.time = t
	solve(Form == 0, unkn, bc, J=Gain, \
		solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-10} }, \
		form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}  )

	phi_.assign(unkn.split(deepcopy=True)[0])
	A_.assign(unkn.split(deepcopy=True)[1])
	u_.assign(unkn.split(deepcopy=True)[2])
	T_.assign(unkn.split(deepcopy=True)[3])
	E_.assign(project(E,Vector))
	B_.assign(project(B,Vector))

	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('time: %.2e s computed in %.0f h %.0f min %.0f s, max. electric potential: %.2f V, min. electric potential: %.2f' % (t, e_h, e_m, e_s, max(phi_.vector()), min(phi_.vector()) )  )

	file_phi << (phi_, t)
	file_A << (A_, t)
	file_E << (E_,t)
	file_B << (B_,t)
	file_u << (u_, t)
	file_T << (T_, t)

	unkn00.assign(unkn0)
	unkn0.assign(unkn)
	t += Dt
