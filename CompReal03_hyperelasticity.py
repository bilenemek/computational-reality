"""Computational reality 03, example of a nonlinear hyperelastic 
material for silicon gel."""
__author__ = "B. Emek Abali"
__license__  = "GNU LGPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

from dolfin import *
from ufl_legacy import indices
xlength=400.0 #[mm]
ylength=1000.0 #[mm]
zlength=1000.0 #[mm]
mesh = BoxMesh(Point(0, 0, 0), Point(xlength, ylength, zlength), 15, 45, 25)
V = VectorFunctionSpace(mesh, 'CG', 1)
cells = MeshFunction('size_t', mesh, 3)
facets = MeshFunction('size_t', mesh, 2)
dA = Measure('ds', domain=mesh, subdomain_data=facets)
dV = Measure('dx', domain=mesh, subdomain_data=cells)
left = CompiledSubDomain('near(x[0],0) && on_boundary')
right = CompiledSubDomain('near(x[0], length) && on_boundary',length=xlength)
bottom = CompiledSubDomain('near(x[1],0) && on_boundary')
top = CompiledSubDomain('near(x[1],length) && on_boundary',length=ylength)
back = CompiledSubDomain('near(x[2],0) && on_boundary')
front= CompiledSubDomain('near(x[2],length) && on_boundary',length=zlength)
facets.set_all(0)
right.mark(facets, 1)
# like a gel filled in a box
bc1 = DirichletBC(V, (0.0,0.0,0.0), left)
bc2 = DirichletBC(V.sub(1), (0.0), top)
bc3 = DirichletBC(V.sub(2), (0.0), top)
bc4 = DirichletBC(V.sub(1), (0.0), bottom)
bc5 = DirichletBC(V.sub(2), (0.0), bottom)
bc6 = DirichletBC(V.sub(1), (0.0), back)
bc7 = DirichletBC(V.sub(2), (0.0), back)
bc8 = DirichletBC(V.sub(1), (0.0), front)
bc9 = DirichletBC(V.sub(2), (0.0), front)
bc=[bc1,bc2,bc3,bc4,bc5,bc6,bc7,bc8,bc9]
# definition for the variational formulation
du = TrialFunction(V)     # Incremental displacement
del_u = TestFunction(V)   # Test function
u  = Function(V)          # Displacement from previous iteration
# material parameters of a silicone gel TSE3062
rho_0 = 1.1E-9 #in ton/mm3
nu = 0.4 #[-]
E = 89.0 #[MPa]
lambada = Constant(E*nu/(1.0+nu)/(1.0-2.0*nu))
mu = Constant(E/(2.0*(1.0+nu)))
i, j, k, l = indices(4)
delta = Identity(3)

def stored_energy(g_u):
	F = delta + g_u
	return as_tensor(lambada*1.0/2.0*ln(det(F))**2 + mu*(F[j,i]*F[j,i]/2.0 - delta[i,i]/2.0 - ln(det(F)) ), [])

grad_u = variable(grad(u))
stored = stored_energy(grad_u)
# traction vector
distr = Expression('a*exp(b*(x[1]-500.0) + c*(x[1]-500.0)*(x[1]-500.0) )',degree=4,a=-50.0,b=0.0,c=-0.001)
N = FacetNormal(mesh)
tr = distr*N
f = Constant((0.0,0.0,0.0)) # body force
Form = (rho_0*f[i]*del_u[i] - diff(stored,grad_u)[i,j]*grad(del_u)[i,j] )*dV + tr[i]*del_u[i]*dA(1)
Gain = derivative(Form,u,du)
solve(Form == 0, u, bc, J=Gain, \
	solver_parameters={"newton_solver":{"linear_solver": "cg", "preconditioner": "ilu", "relative_tolerance": 1e-3} }, \
	form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}  )
pwd = 'results/CR03/'
file = File(pwd+'displacement.pvd')
file << u
