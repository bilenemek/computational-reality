"""Computational reality 16, polarized material, simulation of a capacitor"""
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
2 1 "boundary_in"
2 2 "boundary_out"
2 3 "metal_air"
2 4 "ptfe_air"
2 5 "ptfe_metal"
2 9 "boundary_air"
3 6 "air"
3 7 "metal"
3 8 "ptfe"
------------------------------
'''
data = 'geo/CR16_geo'
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), data+'.h5', 'r')
hdf.read(mesh, '/mesh', False)
Dim = mesh.geometric_dimension()

cells = MeshFunction('size_t', mesh, Dim)
hdf.read(cells, '/cells')
facets = MeshFunction('size_t', mesh, Dim-1)
hdf.read(facets, '/facets')

cells_mat = MeshFunction('size_t', mesh, Dim, 0)
cells_mat.array()[cells.array()==6] = 1 #air
cells_mat.array()[cells.array()==7] = 2 #copper
cells_mat.array()[cells.array()==8] = 3 #ptfe
if cells_mat.array().any()==0 : print('check the renumbering')

mesh_metal = create_meshview(cells_mat,2)
mesh_ptfe = create_meshview(cells_mat,3)

surfaces = {'boundary_in':1,'boundary_out':2,'cu_air':3,'ptfe_air':4, 'ptfe_cu':5, 'boundary_air':9}

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
levicivita2 = as_matrix([ (0,1,-1) , (-1,0,1) , (1,-1,0) ])
levicivita3 = as_tensor([ ( (0,0,0),(0,0,1),(0,-1,0) ) , ( (0,0,-1),(0,0,0),(1,0,0) ) , ( (0,1,0),(-1,0,0),(0,0,0) ) ])
epsilon = levicivita3

eps_0 = 8.85E-12 #in A s/(V m)
mu_0 = 12.6E-7 #in V s/(A m)

null=1E-20 #for numerical reasons it is not zero

#air
varsigma_air = 3E-15
chi_el_air = null
chi_ma_air = null
mu_r_ma_air = chi_ma_air + 1.

#metal (copper)
varsigma_cu = 58.5E+6 #in S/m or in 1/(Ohm m)
chi_el_cu = null
chi_ma_cu = -1E-5
mu_r_ma_cu = chi_ma_cu + 1.

#Teflon (ptfe) is an insulator
varsigma_ptfe = 1E-25 #in S/m or in 1/(Ohm m)
chi_el_ptfe = 1.0
chi_ma_ptfe = 1E-6
mu_r_ma_ptfe = chi_ma_ptfe + 1.

chi_el = material_coefficient(mesh, cells_mat, [chi_el_air, chi_el_cu, chi_el_ptfe], element=coeff_element)
chi_ma = material_coefficient(mesh, cells_mat, [chi_ma_air, chi_ma_cu, chi_ma_ptfe], element=coeff_element)
mu_r_ma = material_coefficient(mesh, cells_mat, [mu_r_ma_air, mu_r_ma_cu, mu_r_ma_ptfe], element=coeff_element)
varsigma = material_coefficient(mesh, cells_mat, [varsigma_air, varsigma_cu, varsigma_ptfe], element=coeff_element)

tMax = 1.0
Dt = 0.1
t = 0.0

capacitor_1 = Expression('0.1*time', degree=1, time=0)
capacitor_2 = Expression('-0.1*time', degree=1, time=0)
bc01=DirichletBC(Space.sub(0), capacitor_1, facets, surfaces['boundary_in'])
bc02=DirichletBC(Space.sub(0), capacitor_2, facets, surfaces['boundary_out'])
bc03=DirichletBC(Space.sub(0), Constant(0.), facets, surfaces['boundary_air'])
bc04=DirichletBC(Space.sub(1), Constant((0.,0.,0.)), facets, surfaces['boundary_air'])

bc = [bc01,bc02,bc03,bc04]

dunkn = TrialFunction(Space)
test = TestFunction(Space)
del_phi, del_A = split(test)

unkn = Function(Space)
unkn0 = Function(Space)
unkn00 = Function(Space)

unkn_init = Expression(('0.0', '0.0','0.0','0.0'), degree=1)
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
MM =1./mu_0/mu_r_ma*chi_ma*B
J_fr = varsigma*E

F_phi = ( -(mD-mD0)[i]*del_phi.dx(i) - Dt*J_fr[i]*del_phi.dx(i) - Dt*epsilon[i,j,k]*MM[k].dx(j)*del_phi.dx(i) )*(dv(1)+dv(2)+dv(3)) + ( n('+')[i]*Dt*(J_fr('+') - J_fr('-'))[i]*del_phi('+') + n('+')[i]*Dt*epsilon[i,j,k]*(MM('+')[k].dx(j) - MM('-')[k].dx(j))*del_phi('+') )*(di(3)+di(4)+di(5)) 

F_A = (eps_0*(A-2.*A0+A00)[i]/Dt/Dt*del_A[i] + 1./mu_0*A[i].dx(j)*del_A[i].dx(j) -J_fr[i]*del_A[i] - (P-P0)[i]/Dt*del_A[i] + epsilon[i,j,k]*MM[k]*del_A[i].dx(j) )*(dv(1)+dv(2)+dv(3))

Form = F_phi + F_A
Gain = derivative(Form, unkn, dunkn)

pwd='results/CR16/capacitor/'
file_phi_metal = File(pwd+'phi_metal.pvd')
file_phi_ptfe = File(pwd+'phi_ptfe.pvd')
file_E = File(pwd+'E.pvd')
file_B = File(pwd+'B.pvd')


FunctionSpace_metal = FunctionSpace(mesh_metal, 'CG', 1)
FunctionSpace_ptfe = FunctionSpace(mesh_ptfe, 'CG', 1)
phi_metal = Function(FunctionSpace_metal, name='$\phi$ in V')
phi_ptfe = Function(FunctionSpace_ptfe, name='$\phi$ in V')
B_out = Function(Vector, name='$B_i$ in T')
E_out = Function(Vector, name='$E_i$ in V/m')

while t < tMax:
	t += Dt
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('time: %.4f s computed in %.0f h %.0f min %.0f s' % (t, e_h, e_m, e_s )  )
	
	capacitor_1.time = t
	capacitor_2.time = t
	solve(Form == 0, unkn, bc, J=Gain, \
		solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-5} }, \
		form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}  )

	phi_metal.assign(project(unkn.split(deepcopy=True)[0], FunctionSpace_metal))
	file_phi_metal << (phi_metal, t)
	phi_ptfe.assign(project(unkn.split(deepcopy=True)[0], FunctionSpace_ptfe))
	file_phi_ptfe << (phi_ptfe, t)
	B_out.assign(project(B,Vector))
	file_B << (B_out,t)
	E_out.assign(project(E,Vector))
	file_E << (E_out,t)

	unkn00.assign(unkn0)
	unkn0.assign(unkn)

