"""Computational reality 16, polarized material, skin and proximity effect"""
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


'''
------------------------------
Dim / ID / Subdomain name
------------------------------
2 1 "air_boundary"
2 2 "air_copper"
2 3 "in"
2 4 "out"
3 5 "air"
3 6 "copper"
------------------------------
'''
surfaces = {'in':3,'out':4,'air_boundary':1,'air_copper':2}

data = 'geo/CR16_geo_proximity_two_wires'
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), data+'.h5', 'r')
hdf.read(mesh, '/mesh', False)
Dim = mesh.geometric_dimension()

cells = MeshFunction('size_t', mesh, Dim)
hdf.read(cells, '/cells')
facets = MeshFunction('size_t', mesh, Dim-1)
hdf.read(facets, '/facets')

cells_mat = MeshFunction('size_t', mesh, Dim, 0)
cells_mat.array()[cells.array()==5] = 1 #air
cells_mat.array()[cells.array()==6] = 2 #copper
if cells_mat.array().any()==0 : print('check the renumbering')


n = FacetNormal(mesh)
#interface, area, volume elements
di = Measure('dS', domain=mesh, subdomain_data=facets)
da = Measure('ds', domain=mesh, subdomain_data=facets)
dv = Measure('dx', domain=mesh, subdomain_data=cells_mat)

Scalar = FunctionSpace(mesh, 'CG', 1)
Vector = VectorFunctionSpace(mesh, 'CG', 1)
Tensor = TensorFunctionSpace(mesh, 'CG', 1)
# phi, A
scalar = FiniteElement("CG", mesh.ufl_cell(), 1) 
vector = VectorElement("CG", mesh.ufl_cell(), 1) 
coeff_element = FiniteElement('DG', mesh.ufl_cell(), 0)
Space = FunctionSpace( mesh, MixedElement([scalar, vector]) )

#units: m, kg, s, A, V, K
delta = Identity(3)
epsilon = as_tensor([ ( (0,0,0),(0,0,1),(0,-1,0) ) , ( (0,0,-1),(0,0,0),(1,0,0) ) , ( (0,1,0),(-1,0,0),(0,0,0) ) ])

eps_0 = 8.85E-12 #in A s/(V m)
mu_0 = 12.6E-7 #in V s/(A m)
null=1E-20 #for numerical reasons it is not zero

#air
varsigma_air = 3E-15
chi_el_air = null
chi_ma_air = null
mu_r_ma_air = chi_ma_air + 1.

#copper
rho_cu = 8960. #in kg / m^3
a = pi*0.01**2/2.
l = 0.1
R = 100.
varsigma_cu = l/(R*a) #in S/m or in 1/(Ohm m)
V = 220. #in V, I = V/R in A
chi_el_cu = null
chi_ma_cu = -1E-5
mu_r_ma_cu = chi_ma_cu + 1.

chi_el = material_coefficient(mesh, cells_mat, [chi_el_air, chi_el_cu], element=coeff_element)
chi_ma = material_coefficient(mesh, cells_mat, [chi_ma_air, chi_ma_cu], element=coeff_element)
mu_r_ma = material_coefficient(mesh, cells_mat, [mu_r_ma_air, mu_r_ma_cu], element=coeff_element)
varsigma = material_coefficient(mesh, cells_mat, [varsigma_air, varsigma_cu], element=coeff_element)

freq = 500000. #in Hz
tMax = 1./freq
Dt = tMax/20.
t = 0.0

bc01=DirichletBC(Space.sub(0), Constant(0.), facets, surfaces['air_boundary'])
bc02=DirichletBC(Space.sub(1), Constant((0.,0.,0.)), facets, surfaces['air_boundary'])
bc_in = Expression('A*sin(nu*2.0*pi*time)',degree=4,A=V/2.,nu=freq,time=0.)
bc_out = Expression('A*sin(nu*2.0*pi*time)',degree=4,A=-V/2.,nu=freq,time=0.)
bc03=DirichletBC(Space.sub(0), bc_in, facets, surfaces['in'])
bc04=DirichletBC(Space.sub(0), bc_out, facets, surfaces['out'])
bc = [bc01,bc02,bc03,bc04]

dunkn = TrialFunction(Space)
test = TestFunction(Space)
del_phi, del_A = split(test)

unkn = Function(Space)
unkn0 = Function(Space)
unkn00 = Function(Space)

unkn_init = Expression(('0.0', '0.0','0.0','0.0'),degree=1)
unkn00 = interpolate(unkn_init,Space)
unkn0.assign(unkn00)
unkn.assign(unkn0)

phi, A = split(unkn)
phi0, A0 = split(unkn0)
phi00, A00 = split(unkn00)

i, j, k, l = indices(4) 
delta = Identity(3)
E = as_tensor(-phi.dx(i)-(A-A0)[i]/Dt, (i,))
E0 = as_tensor(-phi0.dx(i)-(A0-A00)[i]/Dt, (i,))
B = as_tensor(epsilon[i,j,k]*A[k].dx(j), (i,))

D = eps_0*E
D0 = eps_0*E0
H = 1./mu_0*B
P = eps_0*chi_el*E
P0 = eps_0*chi_el*E0
mD = D + P
mD0 = D0 + P0
MM = 1./mu_0/mu_r_ma*chi_ma*B
J_fr = varsigma*E

F_phi = ( -(mD-mD0)[i]*del_phi.dx(i) - Dt*J_fr[i]*del_phi.dx(i) - Dt*epsilon[i,j,k]*MM[k].dx(j)*del_phi.dx(i) )*dv + ( n('+')[i]*Dt*(J_fr('+') - J_fr('-'))[i]*del_phi('+') + n('+')[i]*Dt*epsilon[i,j,k]*(MM('+')[k].dx(j) - MM('-')[k].dx(j))*del_phi('+') )*di(surfaces['air_copper'])

F_A = (eps_0*(A-2.*A0+A00)[i]/Dt/Dt*del_A[i] + 1./mu_0*A[i].dx(j)*del_A[i].dx(j) -J_fr[i]*del_A[i] - (P-P0)[i]/Dt*del_A[i] + epsilon[i,j,k]*MM[k]*del_A[i].dx(j))*dv

Form = F_phi + F_A
Gain = derivative(Form, unkn, dunkn)

pwd='results/CR16/proximity_two_wires/'
file_phi = File(pwd+'phi.pvd')
file_z = File(pwd+'z.pvd')
file_B = File(pwd+'B.pvd')
file_J_fr = File(pwd+'J_fr.pvd')

mesh_1 = create_meshview(cells_mat,1)
mesh_2 = create_meshview(cells_mat,2)

phi_copper_ = Function(FunctionSpace(mesh_2, 'CG', 1), name='$\phi$ in V')
z_ = Function(FunctionSpace(mesh_2, 'CG', 1), name='$z$ in C/kg')
B_ = Function(VectorFunctionSpace(mesh, 'CG', 1), name='$|B_i|$ in T')
J_fr_ = Function(VectorFunctionSpace(mesh_2, 'CG', 1), name='$J_i^\mathrm{fr.}$ in A/m$^2$')


while t < tMax:
	t += Dt
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('time: %.2e s computed in %.0f h %.0f min %.0f s' % (t, e_h, e_m, e_s )  )

	bc_in.time = t
	bc_out.time = t
	solve(Form == 0, unkn, bc, J=Gain, \
		solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-10} }, \
		form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}  )

	phi_copper_.assign(project(unkn.split(deepcopy=True)[0], FunctionSpace(mesh_2, 'CG', 1)))
	file_phi << (phi_copper_, t)
	B_.assign(project(B, VectorFunctionSpace(mesh, 'CG', 1)))
	file_B << (B_,t)
	z_.assign(project(D[i].dx(i)/rho_cu, FunctionSpace(mesh_2, 'CG', 1)))
	file_z << (z_,t)
	J_fr_.assign(project(J_fr, VectorFunctionSpace(mesh_2, 'CG', 1)))
	file_J_fr << (J_fr_,t)

	unkn00.assign(unkn0)
	unkn0.assign(unkn)

