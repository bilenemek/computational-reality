"""Computational reality 16, polarized material, simulation of a transformer"""
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
2 1 "copper_2_out"
2 2 "copper_1_in"
2 3 "copper_1_out"
2 4 "air_goes"
2 5 "copper1_air"
2 6 "copper2_air"
2 11 "air_boundary"
2 12 "copper_2_in"
3 7 "copper_1_with_3_turns"
3 8 "copper_2_with_2_turns"
3 9 "air"
3 10 "goes"
------------------------------

'''
surfaces = {'cu1_in':2,'cu1_out':3,'cu2_in':12,'cu2_out':1,'air_boundary':11,'air_goes':4,'copper1_air':5,'copper2_air':6}


data = 'geo/CR16_geo_trafo'
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), data+'.h5', 'r')
hdf.read(mesh, '/mesh', False)
Dim = mesh.geometric_dimension()

cells = MeshFunction('size_t', mesh, Dim)
hdf.read(cells, '/cells')
facets = MeshFunction('size_t', mesh, Dim-1)
hdf.read(facets, '/facets')

cells_mat = MeshFunction('size_t', mesh, Dim, 0)
cells_mat.array()[cells.array()==7] = 1 #cu
cells_mat.array()[cells.array()==8] = 2 #cu
cells_mat.array()[cells.array()==9] = 3 #air
cells_mat.array()[cells.array()==10] = 4 #goes
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
epsilon = as_tensor([ ( (0,0,0),(0,0,1),(0,-1,0) ) , ( (0,0,-1),(0,0,0),(1,0,0) ) , ( (0,1,0),(-1,0,0),(0,0,0) ) ])
delta = Identity(3)

eps_0 = 8.85E-12 #in A s/(V m)
mu_0 = 12.6E-7 #in V s/(A m)

null=1E-20 #for numerical reasons it is not zero

#Grain Oriented Electrical Steel (GOES) is a ferromagnetic material
varsigma_goes = 2.1E+6 #in S/m or in 1/(Ohm m)
chi_el_goes = null
mu_r_ma_goes = 20000. #approximately
chi_ma_goes = mu_r_ma_goes - 1.

#air
varsigma_air = 3E-15
chi_el_air = null
chi_ma_air = null
mu_r_ma_air = chi_ma_air + 1.

#metal (copper)
a = numpy.pi*0.004**2/2.
l_w = 2.*numpy.pi*0.02
#winding 1 with 3 turns
l_1 = 3.*l_w
R_1 = 100. #in Ohm or 1/S
V_1 = 220. #in V
I_1 = V_1/R_1
varsigma_cu_1 = l_1/(R_1*a) #in S/m
#winding 2 with 2 turns
l_2 = 2.*l_w
R_2 = 45.
varsigma_cu_2 = l_2/(R_2*a) #in S/m or in 1/(Ohm m)

chi_el_cu = null
chi_ma_cu = -1E-5
mu_r_ma_cu = chi_ma_cu + 1.

chi_el = material_coefficient(mesh, cells_mat, [chi_el_cu, chi_el_cu, chi_el_air, chi_el_goes], element=coeff_element)
chi_ma = material_coefficient(mesh, cells_mat, [chi_ma_cu, chi_ma_cu, chi_ma_air, chi_ma_goes], element=coeff_element)
mu_r_ma = material_coefficient(mesh, cells_mat, [mu_r_ma_cu, mu_r_ma_cu, mu_r_ma_air, mu_r_ma_goes], element=coeff_element)
varsigma = material_coefficient(mesh, cells_mat, [varsigma_cu_1, varsigma_cu_2, varsigma_air, varsigma_goes], element=coeff_element)

tMax = 0.02
Dt = tMax/20.
t = 0.0

bc01=DirichletBC(Space.sub(0), Constant(0.), facets, surfaces['air_boundary'])
bc02=DirichletBC(Space.sub(1), Constant((0.,0.,0.)), facets, surfaces['air_boundary'])
bc_cu_in = Expression('A*sin(nu*2.0*pi*time)',degree=1,A=V_1/2.,nu=50.,time=0.)
bc_cu_out = Expression('A*sin(nu*2.0*pi*time)',degree=1,A=-V_1/2.,nu=50.,time=0.)
bc03=DirichletBC(Space.sub(0), bc_cu_in, facets, surfaces['cu1_in'])
bc04=DirichletBC(Space.sub(0), bc_cu_out, facets, surfaces['cu1_out'])
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


F_phi = ( -(mD-mD0)[i]*del_phi.dx(i) - Dt*J_fr[i]*del_phi.dx(i) - Dt*epsilon[i,j,k]*MM[k].dx(j)*del_phi.dx(i) )*dv + n('+')[i]*Dt*epsilon[i,j,k]*(MM('+')[k].dx(j) - MM('-')[k].dx(j))*del_phi('+')*(di(surfaces['cu2_in'])+di(surfaces['cu2_out'])+di(surfaces['air_goes'])+di(surfaces['copper1_air'])+di(surfaces['copper2_air']))

F_A = (eps_0*(A-2.*A0+A00)[i]/Dt/Dt*del_A[i] + 1./mu_0*A[i].dx(j)*del_A[i].dx(j) -J_fr[i]*del_A[i] - (P-P0)[i]/Dt*del_A[i] + epsilon[i,j,k]*MM[k]*del_A[i].dx(j))*dv

Form = F_phi + F_A
Gain = derivative(Form, unkn, dunkn)

pwd='results/CR16/transformer/'
file_phi = File(pwd+'phi.pvd')
file_M = File(pwd+'M.pvd')
file_B = File(pwd+'B.pvd')

mesh_1 = create_meshview(cells_mat,1)
mesh_2 = create_meshview(cells_mat,2)
mesh_3 = create_meshview(cells_mat,3)
mesh_4 = create_meshview(cells_mat,4)

phi_copper_1_ = Function(FunctionSpace(mesh_1, 'CG', 1), name='$\phi$ in V')
phi_copper_2_ = Function(FunctionSpace(mesh_2, 'CG', 1), name='$\phi$ in V')
MM_goes_ = Function(VectorFunctionSpace(mesh_4, 'CG', 1), name='$\|\mathcal{M}_i\|$ in A/m')
B_ = Function(VectorFunctionSpace(mesh, 'CG', 1), name='$|B_i|$ in T')


while t < tMax:
	t += Dt
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('time: %.2e s computed in %.0f h %.0f min %.0f s' % (t, e_h, e_m, e_s )  )

	bc_cu_in.time = t
	bc_cu_out.time = t
	solve(Form == 0, unkn, bc, J=Gain, \
		solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-10} }, \
		form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}  )

	phi_copper_1_.assign(project(unkn.split(deepcopy=True)[0], FunctionSpace(mesh_1, 'CG', 1)))
	file_phi << (phi_copper_1_, t)
	phi_copper_2_.assign(project(unkn.split(deepcopy=True)[0], FunctionSpace(mesh_2, 'CG', 1)))
	file_phi << (phi_copper_2_, t)
	MM_goes_.assign(project(MM, VectorFunctionSpace(mesh_4, 'CG', 1)))
	file_M << (MM_goes_,t)
	B_.assign(project(B, VectorFunctionSpace(mesh, 'CG', 1)))
	file_B << (B_,t)

	unkn00.assign(unkn0)
	unkn0.assign(unkn)

