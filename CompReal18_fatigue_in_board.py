"""Computational reality 18, fatigue in a circuit board"""
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
2 3 "board_air"
2 4 "via_board"
2 5 "voltage_in"
2 6 "voltage_out"
2 7 "clamped"
2 8 "via_air"
3 1 "board"
3 2 "via"
------------------------------
'''
surfaces = {'volt_in':5,'volt_out':6,'clamp':7,'via_air':8,'via_board':4,'board_air':3}

data = 'geo/CR18_board'
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), data+'.h5', 'r')
hdf.read(mesh, '/mesh', False)
Dim = mesh.geometric_dimension()

cells = MeshFunction('size_t', mesh, Dim)
hdf.read(cells, '/cells')
facets = MeshFunction('size_t', mesh, Dim-1)
hdf.read(facets, '/facets')

cells_mat = MeshFunction('size_t', mesh, Dim, 0)
cells_mat.array()[cells.array()==1] = 1 #board
cells_mat.array()[cells.array()==2] = 2 #via
if cells_mat.array().any()==0 : print('check the renumbering')


N = FacetNormal(mesh)
dI = Measure('dS', domain=mesh, subdomain_data=facets)
dA = Measure('ds', domain=mesh, subdomain_data=facets)
dV = Measure('dx', domain=mesh, subdomain_data=cells_mat)

Scalar = FunctionSpace(mesh, 'CG', 1)
Vector = VectorFunctionSpace(mesh, 'CG', 1)
Tensor = TensorFunctionSpace(mesh, 'CG', 1)
#phi, u, T
scalar = FiniteElement("CG", mesh.ufl_cell(), 1) 
vector = VectorElement("CG", mesh.ufl_cell(), 1) 
coeff_element = FiniteElement('DG', mesh.ufl_cell(), 0)
Space = FunctionSpace( mesh, MixedElement([scalar, vector, scalar]) )

mesh_board = create_meshview(cells_mat, 1)
mesh_via = create_meshview(cells_mat, 2)
Scalar_board = FunctionSpace(mesh_board, 'CG', 1)
Scalar_via = FunctionSpace(mesh_via, 'CG', 1)
Vector_via = VectorFunctionSpace(mesh_via, 'CG', 1)
Tensor_via = TensorFunctionSpace(mesh_via, 'CG', 1)

delta = Identity(3)
epsilon = as_tensor([ ( (0,0,0),(0,0,1),(0,-1,0) ) , ( (0,0,-1),(0,0,0),(1,0,0) ) , ( (0,1,0),(-1,0,0),(0,0,0) ) ])
T_ref = 300.0 # K
T_amb = T_ref
eps_0 = 8.85E-12 #in A s/(V m)
mu_0 = 12.6E-7 #in V s/(A m)
null = 1E-20

i,j,k,l,m,n,o,p = indices(8)

freq = 10.
cycle = 1.
tMax = cycle/freq
Dt = tMax/(cycle*10.)
t = 0.0

bc_in = Expression('amp*sin(2.0*pi*nu*time)', degree=4, amp=12., nu=freq, time=0)
bc01=DirichletBC(Space.sub(0), bc_in, facets, surfaces['volt_in'])
bc02=DirichletBC(Space.sub(0), 0.0, facets, surfaces['volt_out'])
bc03=DirichletBC(Space.sub(1), Constant((0.,0.,0.)), facets, surfaces['clamp'])

bc = [bc01,bc02,bc03]

dunkn = TrialFunction(Space)
test = TestFunction(Space)
unkn = Function(Space)
unkn0 = Function(Space)
unkn00 = Function(Space)

unkn_init = Expression(('0.', '0.','0.','0.', 'T_int'),degree=1,T_int=T_amb)
unkn00 = interpolate(unkn_init,Space)
unkn0.assign(unkn00)
unkn.assign(unkn0)

del_phi,del_u,del_T = split(test)
phi,u,T = split(unkn)
phi0,u0,T0 = split(unkn0)
phi00,u00,T00 = split(unkn00)

def VoigtToTensor(A):
	A11, A12, A13, A14, A15, A16 = A[0,0], A[0,1], A[0,2], A[0,3], A[0,4], A[0,5]
	A22, A23, A24, A25, A26 = A[1,1], A[1,2], A[1,3], A[1,4], A[1,5]
	A33, A34, A35, A36 = A[2,2], A[2,3], A[2,4], A[2,5]
	A44, A45, A46 = A[3,3], A[3,4], A[3,5]
	A55, A56 = A[4,4], A[4,5]
	A66 = A[5,5]
	A21, A31, A41, A51, A61 = A12, A13, A14, A15, A16
	A32, A42, A52, A62 = A23, A24, A25, A26
	A43, A53, A63 = A34, A35, A36
	A54, A64 = A45, A46
	A65 = A56
	return as_tensor([ \
	[ \
	[ [A11,A16,A15], [A16,A12,A14], [A15,A14,A13]] , \
	[ [A61,A66,A65], [A66,A62,A64], [A65,A64,A63]] , \
	[ [A51,A56,A55], [A56,A52,A54], [A55,A54,A53]] \
	] , [ \
	[ [A61,A66,A65], [A66,A62,A64], [A65,A64,A63]] , \
	[ [A21,A26,A25], [A26,A22,A24], [A25,A24,A23]] , \
	[ [A41,A46,A45], [A46,A42,A44], [A45,A44,A43]] \
	] , [ \
	[ [A51,A56,A55], [A56,A52,A54], [A55,A54,A53]] , \
	[ [A41,A46,A45], [A46,A42,A44], [A45,A44,A43]] , \
	[ [A31,A36,A35], [A36,A32,A34], [A35,A34,A33]] ] \
	])

#laminate materials data (for board)
varsigma_l = null #in S/m
pi_l = null #V/K
kappa_l =  1.3 #in W/(m K)
capacity_l = 800. #in J / (kg K)
alp11_l = 13.2E-6 #in 1/K
alp22_l = 16.7E-6 #in 1/K
alp33_l = 39E-6 #in 1/K
alpha_l = as_tensor([ [alp11_l,0.,0.],[0.,alp22_l,0.],[0.,0.,alp33_l] ])
h_l = 10. #in J / (s m^2 K)
rho_l = 2.5E3 #in kg/m^3
C_voigt_l = numpy.array([ \
[66242.E6, 41797.E6, 37814.E6, 0, 0, 0],\
[41797.E6, 50460.E6, 32290.E6, 0, 0, 0],\
[37814.E6, 32290.E6, 31591.E6, 0, 0, 0],\
[0, 0, 0, 2250.E6, 0, 0],\
[0, 0, 0, 0, 2250.E6, 0],\
[0, 0, 0, 0, 0, 6630.E6]  ])  #in Pa
C_l = VoigtToTensor(C_voigt_l)

#copper data for a via in Printed Circuit Boards (PCBs)
varsigma_cu = 1.7E4 #in S/m
pi_cu = 1.8E-6 #V/K
kappa_cu = 385. #in W/(m K)
capacity_cu = 390. #in J / (kg K)
alp_cu = 17E-6 #in 1/K
alpha_cu = as_tensor([ [alp_cu,0.,0.],[0.,alp_cu,0.],[0.,0.,alp_cu] ])
h_cu = 10. #in J / (s m^2 K)
rho_cu = 8.94E3 #kg/m^3
sigmaY_cu = 100.E6 #in Pa
h_pl_cu = 615.E6 #in Pa
p1,p2,p3=169100.E6,122200.E6,75420.E6 #in Pa
C_voigt_cu = numpy.array([ \
[p1, p2, p2, 0, 0, 0],\
[p2, p1, p2, 0, 0, 0],\
[p2, p2, p1, 0, 0, 0],\
[0, 0, 0, p3, 0, 0],\
[0, 0, 0, 0, p3, 0],\
[0, 0, 0, 0, 0, p3]  ])
C_cu = VoigtToTensor(C_voigt_cu)

rho = material_coefficient(mesh, cells_mat, [rho_l, rho_cu], element=coeff_element)
varsigma = material_coefficient(mesh, cells_mat, [varsigma_l, varsigma_cu], element=coeff_element)
pi_ = material_coefficient(mesh, cells_mat, [pi_l, pi_cu], element=coeff_element)
kappa = material_coefficient(mesh, cells_mat, [kappa_l, kappa_cu], element=coeff_element)
capacity = material_coefficient(mesh, cells_mat, [capacity_l, capacity_cu], element=coeff_element)
h = material_coefficient(mesh, cells_mat, [h_l, h_cu], element=coeff_element)

eps = sym(grad(u))
eps0 = sym(grad(u0))
epsDot = (eps-eps0)/Dt
v = as_tensor( (u-u0)[i]/Dt, (i,))
v0 = as_tensor( (u0-u00)[i]/Dt, (i,))
E = -grad(phi)
E0 = -grad(phi0)
EE = E
D = eps_0*E
D0 = eps_0*E0
JJ = as_tensor( varsigma*pi_*T.dx(i) + varsigma*EE[i], (i,))
J = as_tensor( D[j].dx(j)*v[i] + JJ[i], (i,))
FF = as_tensor(D[j].dx(j)*E[i], (i,))
q = as_tensor(-kappa*T.dx(i)+varsigma*pi_*T*EE[i], (i,))
Phi = q/T

f = Constant((0.,0.,0.))
r = Constant(0.0)

#laminate
peps_eq_l = Function(Scalar_board)
teps_l = as_tensor( alpha_l[i,j]*(T-T_ref), (i,j))
sigma_l = as_tensor( C_l[i,j,k,l]*(eps[k,l] - teps_l[k,l]), (i,j))
eta_l = as_tensor( capacity*ln(T/T_ref) + alpha_l[i,j]/rho*sigma_l[i,j], ())
eta0_l = as_tensor( capacity*ln(T0/T_ref) + C_l[i,j,k,l]*alpha_l[k,l]/rho*eps0[i,j], ())
Sigma_l = as_tensor(-q[i]/T/T*T.dx(i) + JJ[i]/T*EE[i] , ())

#copper
gamma_cu = Function(Scalar)
flow_cond_0 = Function(Scalar)
direction_cond_0 = Function(Scalar)
sigma0_cu = Function(Tensor)
peps0_cu = Function(Tensor)
beta0_cu = Function(Tensor)
peps_eq_cu = Function(Scalar_via)

dev_sigma0_cu = as_tensor(sigma0_cu[i,j]-1./3.*sigma0_cu[k,k]*delta[i,j] , (i,j))
stress_diff = dev_sigma0_cu-beta0_cu
LambdaDot_cu = as_tensor( (stress_diff[i,j]*C_cu[i,j,k,l]*epsDot[k,l])   / (4./9.*sigmaY_cu**2*h_pl_cu + stress_diff[i,j]*C_cu[i,j,k,l]*stress_diff[k,l] ) , ()) # epsDot / Pa

flow_cond = as_tensor(1./2.*stress_diff[i,j]*stress_diff[i,j] - 1./3.*sigmaY_cu**2, ())
direction_cond = as_tensor( stress_diff[i,j]*epsDot[i,j], ())

s_type = 'mumps'
f_param = {"cpp_optimize": True, "representation":"uflacs",  "quadrature_degree": 2}

assign(flow_cond_0, project(flow_cond, Scalar, solver_type=s_type, form_compiler_parameters=f_param))    
assign(direction_cond_0, project(direction_cond, Scalar, solver_type=s_type, form_compiler_parameters=f_param))
gamma_ = Expression ("flow_cond >= 0 && direction > 0 ? 1: 0", flow_cond = flow_cond_0, direction = direction_cond_0, degree = 3)

pepsDot_cu = gamma_*LambdaDot_cu*stress_diff
teps_cu = as_tensor( alpha_cu[i,j]*(T-T_ref), (i,j))
tepsDot_cu = as_tensor( alpha_cu[i,j]*(T-T0)/Dt, (i,j))
sigmaDot_cu = as_tensor( C_cu[i,j,k,l]*(epsDot-tepsDot_cu-pepsDot_cu)[k,l], (i,j))

betaDot_cu = 2./3.*h_pl_cu*pepsDot_cu

sigma_cu = sigma0_cu + Dt*sigmaDot_cu
beta_cu = beta0_cu + Dt*betaDot_cu
peps_cu = peps0_cu + Dt*pepsDot_cu

eta_cu = as_tensor( capacity*ln(T/T_ref) + alpha_cu[i,j]/rho*sigma_cu[i,j], ())
eta0_cu = as_tensor( capacity*ln(T0/T_ref) + alpha_cu[i,j]/rho*sigma0_cu[i,j], ())
Sigma_cu = as_tensor(-q[i]/T/T*T.dx(i) + JJ[i]/T*EE[i] + sigma_cu[i,j]/T*peps_cu[i,j] , ())


F_phi = (-(D-D0)[i]*del_phi.dx(i) - Dt*J[i]*del_phi.dx(i) )*(dV(1)+dV(2)) + N('+')[i]*Dt*(J('+')-J('-'))[i]*del_phi('+')*dI(surfaces['via_board']) + N[i]*Dt*J[i]*del_phi*(dA(surfaces['volt_in'])+dA(surfaces['volt_out'])+dA(surfaces['clamp'])+dA(surfaces['via_air'])+dA(surfaces['board_air']))

F_u = (rho*(v-v0)[i]/Dt*del_u[i]  - rho*f[i]*del_u[i] - FF[i]*del_u[i] )*(dV(1)+dV(2)) + sigma_l[j,i]*del_u[i].dx(j)*dV(1) + sigma_cu[j,i]*del_u[i].dx(j)*dV(2)

F_T = ( - Dt*Phi[i]*del_T.dx(i) - Dt*rho*r/T*del_T )*(dV(1)+dV(2)) + (rho*(eta_l-eta0_l)*del_T - Dt*Sigma_l*del_T )*dV(1) + (rho*(eta_cu-eta0_cu)*del_T - Dt*Sigma_cu*del_T )*dV(2) + Dt*h*(T-T_ref)/T*del_T*(dA(surfaces['volt_in'])+dA(surfaces['volt_out'])+dA(surfaces['clamp'])+dA(surfaces['via_air'])+dA(surfaces['board_air']))


Form = F_phi + F_u + F_T
Gain = derivative(Form, unkn, dunkn)

pwd='results/CR18/'
file_phi_via = File(pwd+'phi.pvd')
file_J_via = File(pwd+'J.pvd')
file_u = File(pwd+'u.pvd')
file_peps_via = File(pwd+'peps_via.pvd')
file_peps_board = File(pwd+'peps_board.pvd')
file_T = File(pwd+'T.pvd')

phi_via_ = Function(Scalar_via, name='$\phi$ in V')
J_via_ = Function(Vector_via, name='$|J_i|$ in A/m$^2$')
u_ = Function(Vector, name='$|u_i|$')
peps_via_ = Function(Scalar_via, name='$^\mathrm{p}\!\varepsilon_\mathrm{eq.}$')
T_ = Function(Scalar, name='$T$ in K')

while t < tMax:

	bc_in.time = t
	solve(Form == 0, unkn, bc, J=Gain, \
		solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-5} }, \
		form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}  )

	phi_via_.assign(project(unkn.split(deepcopy=True)[0],Scalar_via))
	J_via_.assign(project(J,Vector_via))
	u_.assign(unkn.split(deepcopy=True)[1])
	T_.assign(unkn.split(deepcopy=True)[2])
	file_phi_via << (phi_via_, t)
	file_J_via << (J_via_, t)
	file_u << (u_, t)
	file_T << (T_, t)

	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('time: %.2e s computed in %.0f h %.0f min %.0f s, T max = %.2f K ' % (t, e_h, e_m, e_s, max(unkn.split(deepcopy=True)[2].vector()) )  )

	sigma_cu_ = project(sigma_cu, Tensor, solver_type="mumps", \
		form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2})
	sigma0_cu.assign(sigma_cu_)

	beta_cu_ = project(beta_cu, Tensor, solver_type="mumps", \
		form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2})
	beta0_cu.assign(beta_cu_)
	
	flow_cond = as_tensor(1./2.*stress_diff[i,j]*stress_diff[i,j] - 1./3.*sigmaY_cu**2, ())
	direction_cond = as_tensor( stress_diff[i,j]*epsDot[i,j], ())

	assign(flow_cond_0, project(flow_cond, Scalar, solver_type=s_type, form_compiler_parameters=f_param))    
	assign(direction_cond_0, project(direction_cond, Scalar, solver_type=s_type, form_compiler_parameters=f_param))
	gamma_ = Expression ("flow_cond >= 0 && direction > 0 ? 1: 0", flow_cond = flow_cond_0, direction = direction_cond_0, degree = 3)
	
	pepsDot_cu = gamma_*LambdaDot_cu*stress_diff
	sigmaDot_cu = as_tensor( C_cu[i,j,k,l]*(epsDot-tepsDot_cu-pepsDot_cu)[k,l], (i,j))

	betaDot_cu = 2./3.*h_pl_cu*pepsDot_cu

	sigma_cu = sigma0_cu + Dt*sigmaDot_cu
	beta_cu = beta0_cu + Dt*betaDot_cu
	peps_cu = peps0_cu + Dt*pepsDot_cu
	
	peps_eq_ = project(peps_eq_cu + Dt*(2./3.*pepsDot_cu[i,j]*pepsDot_cu[i,j])**0.5, \
		Scalar_via, solver_type="mumps", form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2})
	print( 'Plasticity: ',max(gamma_cu.vector()) )
	peps_eq_cu.assign(peps_eq_)
	file_peps_via << (peps_eq_cu,t)
	file_peps_board << (peps_eq_l,t)

	unkn00.assign(unkn0)
	unkn0.assign(unkn)
	t += Dt
