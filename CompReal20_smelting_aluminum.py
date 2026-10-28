"""Computational reality 20, electric smelting"""
__author__ = "B. Emek Abali"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

import time as comp_time
start_time = comp_time.time()

from dolfin import *
from ufl_legacy import indices
import numpy
#set_log_level(50)

class material_coefficient(UserExpression):
	def __init__(self, target_mesh, use_list, with_coeffs, **kwargs):
		super().__init__(**kwargs)
		self._mesh = target_mesh
		self.cell_list = use_list
		self.coeffs = with_coeffs
		self._element = kwargs["element"]
	def eval_cell(self, values, x, cell):
		values[0]= self.coeffs[ self.cell_list[cell.index] -1 ]

processID =  MPI.comm_world.Get_rank()
#units: m, kg, s, V, K
delta = Identity(3)
epsilon = as_tensor([ ( (0,0,0),(0,0,1),(0,-1,0) ) , ( (0,0,-1),(0,0,0),(1,0,0) ) , ( (0,1,0),(-1,0,0),(0,0,0) ) ])

t = 0.0
tMax = 1000.
Dt = 50.

'''
------------------------------
Dim / ID / Subdomain name
------------------------------
2 1 "anode"
2 3 "cathode"
2 4 "walls"
2 5 "top"
3 2 "cyrolite"
------------------------------
'''
surfaces = {'anode':1,'cathode':3,'walls':4,'top':5}


mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), 'geo/CR20_smelting_aluminum.h5', 'r')
hdf.read(mesh, '/mesh', False)
Dim = mesh.geometric_dimension()

cells = MeshFunction('size_t', mesh, Dim)
hdf.read(cells, '/cells')
facets = MeshFunction('size_t', mesh, Dim-1)
hdf.read(facets, '/facets')

Scalar = FunctionSpace(mesh, 'CG', 1)
Vector = VectorFunctionSpace(mesh, 'CG', 1)
scalar = FiniteElement("CG", mesh.ufl_cell(), 1) 
vector = VectorElement("CG", mesh.ufl_cell(), 1)
Space = FunctionSpace( mesh, MixedElement([scalar, vector, scalar, vector, scalar]) ) #phi, A, p, v, T
da = Measure('ds', domain=mesh, subdomain_data=facets)
dv = Measure('dx', domain=mesh, subdomain_data=cells)

n = FacetNormal(mesh)

dunkn = TrialFunction(Space)
test = TestFunction(Space)
unkn = Function(Space)
unkn0 = Function(Space)
unkn00 = Function(Space)

del_phi,del_A,del_p,del_v,del_T = split(test)
phi,A,p,v,T = split(unkn)

f = Constant((0.0, 0.0, 0.0))
r = Constant(0.0)
Tref = 960. #in K
pref = 1E5 #in Pa
Tamb = Tref
eps_0 = 8.85E-12 #in A s/(V m)
mu_0 = 12.6E-7 #in V s/(A m)
h = 0.1 #in J / (s m^2 K)

varsigma = 3.95E2*exp(-1192./T)
pi = -1.28E-3 #V/K 
kappa = 0.8 #in W/(m K)
chi_mag = 2.2E-5
mu_mag_rel = chi_mag + 1.
cp = 394.7/(209.9413E-3) #in J/(kg K)
c4 = 2.259E-7 #in m3/(kg K)
vol_ref = 1.891E-4 #in m3/kg
c5 = 1E2
mu = 2.5E-2 #in Pa s
la = mu*1E3 #in Pa s

#boundary conditions------------------------------------------------
phi_in = Expression('4.0E-3*t', degree=1, t=0. )
phi_out = Expression('0.0',degree=1)
bc = []
#electric potential on anode and cathode
bc.append( DirichletBC(Space.sub(0), phi_in, facets, surfaces['anode']) )
bc.append( DirichletBC(Space.sub(0), phi_out, facets, surfaces['cathode']) )
#reference p on the top opening
bc.append( DirichletBC(Space.sub(2), pref, facets, surfaces['top']) )
#zero velocity on the walls of the bath
bc.append( DirichletBC(Space.sub(3).sub(2), Constant(0.0), facets, surfaces['top']) )
bc.append( DirichletBC(Space.sub(3), Constant((0.0,0.0,0.0)), facets, surfaces['walls']) )
#zero velocity on anode, cathode
bc.append( DirichletBC(Space.sub(3), Constant((0.0,0.0,0.0)), facets, surfaces['anode']) )
bc.append( DirichletBC(Space.sub(3), Constant((0.0,0.0,0.0)), facets, surfaces['cathode']) )

#initial values for phi, A, p, v, T  ------------------------------------
unkn_init = Expression(('0.', '0.','0.','0.', 'p_r', '0.','0.','0.', 'T_r'),degree=1, T_r=Tref, p_r = pref)
unkn00 = interpolate(unkn_init,Space)
unkn0.assign(unkn00)
unkn.assign(unkn0)
phi0,A0,p0,v0,T0 = split(unkn0)
phi00,A00,p00,v00,T00 = split(unkn00)

i, j, k, l = indices(4)

#electromagnetic fields ------------------------------------
E = -grad(phi) - (A-A0)/Dt
E0 = -grad(phi0) - (A0-A00)/Dt
B = as_tensor(epsilon[i,j,k]*A[k].dx(j) , (i,))
B0 = as_tensor(epsilon[i,j,k]*A0[k].dx(j) , (i,))
EE = as_tensor( E[i] + epsilon[i,j,k]*v[j]*B[k] , (i,))

#constitutive equations ------------------------------------
D = eps_0*E
D0 = eps_0*E0
H = 1./mu_0*B
H0 = 1./mu_0*B0
m = as_tensor( -1./2.*delta[j,i]*(H[k]*B[k]+D[k]*E[k]) + H[i]*B[j] + D[j]*E[i] , (j,i))

#liquid copper (molten metal) ------------------------------------
mD = D
mD0 = D0
GG = as_tensor( epsilon[i,j,k]*mD[j]*B[k] , (i,))
GG0 = as_tensor( epsilon[i,j,k]*mD0[j]*B0[k] , (i,))
MM = chi_mag/mu_0/mu_mag_rel*B
MM0 = chi_mag/mu_0/mu_mag_rel*B0
dsigma = as_tensor( -MM[i]*B[j] + la*v[k].dx(k)*delta[j,i] + 2.*mu*sym(grad(v))[i,j] , (j,i))
eta = as_tensor(cp*ln(T/Tref) - c4*(p-pref+MM[i]*B[i]) , ())
eta0 = as_tensor(cp*ln(T0/Tref) - c4*(p0-pref+MM0[i]*B0[i]) , ())
vol = as_tensor(vol_ref + c4*(T-Tref) + c5*(p-pref + MM[i]*B[i]) , ())
vol0 = as_tensor(vol_ref + c4*(T0-Tref) + c5*(p0-pref + MM0[i]*B0[i]) , ())
rho = 1./vol
rho0 = 1./vol0
JJ_fr = as_tensor(varsigma*pi*T.dx(i) + varsigma*EE[i] , (i,) ) 
J_fr = as_tensor( JJ_fr[i] + mD[j].dx(j)*v[i] , (i,))
q = as_tensor(-kappa*T.dx(i) + varsigma*pi*EE[i] , (i,))
Phi = q/T
Sigma = as_tensor(-q[i]/T/T*T.dx(i) + 1./T*(dsigma[j,i]+MM[i]*B[j])*v[i].dx(j) + 1./T*JJ_fr[i]*EE[i] , ())

#weak forms-----------------------------------------------------

F_phi = (-(mD-mD0)[i]*del_phi.dx(i) - Dt*J_fr[i]*del_phi.dx(i) )*dv \
+ n[i]*Dt*( J_fr[i] + epsilon[i,j,k]*MM[k].dx(j) )*del_phi*(da(surfaces['walls'])+da(surfaces['top']))

F_A = (eps_0*(A-2.*A0+A00)[i]/Dt/Dt*del_A[i] + 1./mu_0*A[i].dx(j)*del_A[i].dx(j) -J_fr[i]*del_A[i] + epsilon[i,j,k]*MM[k]*del_A[i].dx(j) )*dv

F_p = ( (rho-rho0) + Dt*rho.dx(i)*v[i] + Dt*rho*v[i].dx(i) )*del_p/rho*dv

F_v = ( (rho*v[i] - rho0*v0[i] + GG[i] - GG0[i] )*del_v[i] + Dt*rho.dx(j)*v[i]*v[j]*del_v[i] + Dt*rho*v[i].dx(j)*v[j]*del_v[i] + Dt*rho*v[i]*v[j].dx(j)*del_v[i] + Dt*p.dx(i)*del_v[i] + Dt*(m[j,i]+dsigma[j,i])*del_v[i].dx(j) - Dt*rho*f[i]*del_v[i] )*dv

F_T = ( rho*(eta-eta0)*del_T - Dt*eta*(rho*v[i]*del_T).dx(i) - Dt*Phi[i]*del_T.dx(i) - Dt*rho*r/T*del_T - Dt*Sigma*del_T )*dv \
+ Dt/T*h*(T-Tamb) *del_T*(da(surfaces['anode'])+da(surfaces['cathode'])+da(surfaces['walls'])+da(surfaces['top']))

Form = F_phi + F_A + F_p + F_v + F_T
Gain = derivative(Form, unkn, dunkn)

pwd = 'results/CR20/'
file_phi = File(pwd + 'phi.pvd')
file_A = File(pwd + 'A.pvd')
file_p = File(pwd + 'p.pvd')
file_v = File(pwd + 'v.pvd')
file_T = File(pwd + 'T.pvd')
file_J = File(pwd + 'J.pvd')
file_B = File(pwd + 'B.pvd')


f_param = {"cpp_optimize": True, "representation":"uflacs",  "quadrature_degree": 2}

while t < tMax:
	t += Dt
	phi_in.t = t
	#phi_out.t = t
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('time: %.2e s computed in %.0f h %.0f min %.0f s ' % (t, e_h, e_m, e_s )  )
	
	solve(Form == 0, unkn, bc, J=Gain, \
		solver_parameters={"newton_solver":{"linear_solver": "gmres", "preconditioner": "hypre_euclid", "relative_tolerance": 1e-10} }, \
		form_compiler_parameters=f_param )
	
	unkn00.assign(unkn0)
	unkn0.assign(unkn)

	file_phi << (unkn.split()[0], t)
	file_A << (unkn.split()[1], t)
	file_p << (unkn.split()[2], t)
	file_v << (unkn.split()[3], t)
	file_T << (unkn.split()[4], t)
	file_J << (project(JJ_fr, Vector, solver_type="gmres", form_compiler_parameters=f_param), t)
	file_B << (project(B, Vector, solver_type="gmres", form_compiler_parameters=f_param), t)

