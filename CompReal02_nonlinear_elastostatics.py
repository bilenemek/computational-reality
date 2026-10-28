"""Computational reality 02, example of a nonlinear elastostatics, 
a thin plate is pressed down."""
__author__ = "B. Emek Abali"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

from dolfin import *
from ufl_legacy import indices

xlength=100.0 #in mm
ylength=50.0
zlength=1.0
#origin is in the middle of the plate
mesh = BoxMesh(Point(0,0,0), Point(xlength,ylength,zlength), 100, 50, 1)
V = VectorFunctionSpace(mesh, 'CG', 1)
cells = MeshFunction('size_t', mesh, 3)
facets = MeshFunction('size_t', mesh, 2)
dA = Measure('ds', domain=mesh, subdomain_data=facets)
dV = Measure('dx', domain=mesh, subdomain_data=cells)
left = CompiledSubDomain('near(x[0],0) && on_boundary')
right = CompiledSubDomain('near(x[0],l) && on_boundary',l=xlength)
top = CompiledSubDomain('pow(x[0]-X,2)+pow(x[1]-Y,2)<pow(10.0,2) && near(x[2],l) && on_boundary',X=xlength/4.,Y=ylength/3.,l=zlength)
facets.set_all(0)
top.mark(facets, 1)
tr = Constant((0.0,0.0,200.0)) # MPa
null = Constant((0.0,0.0,0.0))
bc1=DirichletBC(V, null, left)
bc2=DirichletBC(V, null, right)
bc = [bc1,bc2]
# definition for the variational formulation for a nonlinear form
du = TrialFunction(V)            # Incremental displacement
del_u  = TestFunction(V)         # Test function
u  = Function(V)                 # Displacement from previous iteration
# material parameters of a stainless steel
nu = 0.3
E = 210000.0 #MPa
G = E/(2.0*(1.0+nu))
# Lame parameters (lambda has another meaning in python as def, 
# thus it is forbidden to use as a variable)
lambada = 2.0*G*nu / (1.0 - 2.0*nu)
mu = G
# metric
delta = Identity(3)
# index notation
i, j, k, l = indices(4)
# deformation gradient
F = as_tensor(u[i].dx(j) + delta[i,j] , (i,j))
J = det(F)
# right Cauchy-Green deformation tensor
C = as_tensor(F[i,k]*F[i,j] , (k,j))
# Green-Lagrange strain tensor
E = as_tensor(1./2.*( C[k,j] - delta[k,j] ) , (k,j))
# second Piola-Kirchhoff stress tensor
S = as_tensor(lambada*E[l,l]*delta[k,j] + 2.0*mu*E[k,j] , (k,j))
# nominal stress
P = as_tensor(F[i,j]*S[j,k] , (k,i))

Form = P[k,i]*del_u[i].dx(k)*dV - tr[i]*del_u[i]*dA(1)
Gain = derivative(Form, u, du)
solve(Form == 0, u, bc, J=Gain, \
	solver_parameters={"newton_solver":{"linear_solver": "lu", "relative_tolerance": 1e-3} }, \
	form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}  )

# write out
pwd='results/CR02/'
file = File(pwd+'nonlin_elastostatics_deformations.pvd')
file << u
# Cauchy stress tensor
sigma = as_tensor(1./J*F[j,k]*P[k,i] , (j,i))
sigma_dev = as_tensor(sigma[i,j]-1.0/3.0*sigma[k,k]*delta[i,j] , (i,j))
eqStress = as_tensor((3.0/2.0*sigma_dev[i,j]*sigma_dev[i,j])**0.5 , ())
# now we have displacements, u, so we can calculate the equivalent
# stress by projecting to the same space, so it looks cool
eqS = project(eqStress, FunctionSpace(mesh, 'CG', 1))
file = File(pwd+'nonlin_elastostatics_eqStress.pvd')
file << eqS
