"""Computational relativity 01, example of a linear elastostatics, 
a box is clamped on one end and loaded on the other."""
__author__ = "B. Emek Abali"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

# import all packages from fenics into cache
from dolfin import *
from ufl_legacy import indices
# geometry is a 3D box, looks like a beam, with 15x15x75 elements in x,y,z
xlength=500.0 #in mm
ylength=100.0 #in mm
zlength=100.0 #in mm
mesh = BoxMesh(Point(0, 0, 0), Point(xlength, ylength, zlength), 50, 10, 10)
# vector space with polynomial degree 1 for displacement field
V = VectorFunctionSpace(mesh, 'CG', 1)
# element surfaces, facets are 2D surface boundaries for 3D elements (cells)
# or they are 1D line boundaries for 2D elements (cells)
# or they are the edges of 1D elements (cells)
cells = MeshFunction('size_t', mesh,3)
facets = MeshFunction('size_t', mesh,2)
dA = Measure('ds', domain=mesh, subdomain_data=facets)
dV = Measure('dx', domain=mesh, subdomain_data=cells)

# from the domain V find the domains on the surface for clamping and loading
left = CompiledSubDomain('near(x[0], 0) && on_boundary')
right = CompiledSubDomain('near(x[0], length) && on_boundary', length=xlength)
# marking the parts of the surface
facets.set_all(0)
# by integrating dA(1) means a surface integral over the domain 'right'
right.mark(facets, 1)
# traction vector
tr = Expression(('A+ B*x[1] + C*x[2]','0.0','0.0'), A=2.0, B=0.5, C=0.5, degree=1) #in MPa 

# a zero for the left clamped in the wall
null = Constant((0.0,0.0,0.0))
# stating the Dirichlet boundary values, which will be applied after the assembly
bc = [DirichletBC(V, null, left)]
'''
bc = []
bc.append( DirichletBC(V, null, left) )
'''
# definition for the variational formulation, Galerkin method
u = TrialFunction(V)
del_u = TestFunction(V)

# material parameters of an AISI steel
nu = 0.3
E = 210000.0 #in MPa
G = E/(2.0*(1.0+nu))
# Lame parameters
la = 2.0*G*nu / (1.0 - 2.0*nu)
mu = G
# Kronecker delta in 3D
delta = Identity(3)
i,j,k = indices(3)
# strain tensor
eps = as_tensor(1.0/2.0*(u[i].dx(j)+u[j].dx(i)) , (i,j))
# Cauchy stress tensor
sigma = as_tensor(la*eps[k,k]*delta[i,j]+2.0*mu*eps[i,j] , (i,j))
# Variational form
a = sigma[i,j]*del_u[i].dx(j)*dV 
L = tr[i]*del_u[i]*dA(1)
disp = Function(V)
solve(a == L, disp, bcs=bc, solver_parameters={'linear_solver': 'cg'})
# write out
file_ = File('results/CR01/elastostatics.pvd')
file_ << disp
# get the value of deflection on the tip
# print( 'tip deflection reads', disp(xlength,ylength/2.0,zlength/2.0))
