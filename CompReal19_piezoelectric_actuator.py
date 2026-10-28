"""Computational reality 19, piezoelectric transducer"""
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

#units: m, kg, s, V, K
delta = Identity(3)
epsilon = as_tensor([ ( (0,0,0),(0,0,1),(0,-1,0) ) , ( (0,0,-1),(0,0,0),(1,0,0) ) , ( (0,1,0),(-1,0,0),(0,0,0) ) ])

def VoigtToTensorRank4(A11=0., A12=0., A13=0., A14=0., A15=0., A16=0., A22=0., A23=0., A24=0., A25=0., A26=0., A33=0., A34=0., A35=0., A36=0., A44=0., A45=0., A46=0., A55=0., A56=0., A66=0.):
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

def VoigtToTensorRank3(A11=0.,A12=0.,A13=0.,A14=0.,A15=0.,A16=0., A21=0.,A22=0.,A23=0.,A24=0.,A25=0.,A26=0., A31=0.,A32=0.,A33=0.,A34=0.,A35=0.,A36=0.):
	return as_tensor([ \
	[ \
	[ A11, A16, A15 ] , \
	[ A16, A12, A14 ] , \
	[ A15, A14, A13 ] \
	] , [ \
	[ A21, A26, A25 ] , \
	[ A26, A22, A24 ] , \
	[ A25, A24, A23 ] \
	] , [ \
	[ A31, A36, A35 ] , \
	[ A36, A32, A34 ] , \
	[ A35, A34, A33 ] ] \
	])

def ToTensorRank2(A11=0.,A12=0.,A13=0., A21=0.,A22=0.,A23=0., A31=0.,A32=0.,A33=0.):
	A21, A31, A32 = A12, A13, A23
	return as_tensor([ \
	[ A11, A12, A13 ] , \
	[ A21, A22, A23 ] , \
	[ A31, A32, A33 ] \
	] )


'''
------------------------------
Dim / ID / Subdomain name
------------------------------
2 1 "singularity"
2 2 "boundary_air"
2 5 "clamp"
2 6 "loading"
3 3 "brass"
3 4 "piezo"
------------------------------
'''
surfaces = {'singularity':1,'boundary_air':2,'clamp':5,'loading':6}

data = 'geo/CR19_piezo'
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), data+'.h5', 'r')
hdf.read(mesh, '/mesh', False)
Dim = mesh.geometric_dimension()

cells = MeshFunction('size_t', mesh, Dim)
hdf.read(cells, '/cells')
facets = MeshFunction('size_t', mesh, Dim-1)
hdf.read(facets, '/facets')

cells_mat = MeshFunction('size_t', mesh, Dim, 0)
cells_mat.array()[cells.array()==3] = 1 #brass
cells_mat.array()[cells.array()==4] = 2 #piezo
if cells_mat.array().any()==0 : print('check the renumbering')

N = FacetNormal(mesh)

Scalar = FunctionSpace(mesh, 'CG', 1)
Vector = VectorFunctionSpace(mesh, 'CG', 1)
Tensor = TensorFunctionSpace(mesh, 'CG', 1)
#phi, A, u, T
scalar = FiniteElement("CG", mesh.ufl_cell(), 1) 
vector = VectorElement("CG", mesh.ufl_cell(), 1) 
coeff_element = FiniteElement('DG', mesh.ufl_cell(), 0)
Space = FunctionSpace( mesh, MixedElement([scalar, vector, vector, scalar]) )

dI = Measure('dS', domain=mesh, subdomain_data=facets)
dA = Measure('ds', domain=mesh, subdomain_data=facets)
dV = Measure('dx', domain=mesh, subdomain_data=cells_mat)

nu = 10. #in Hz
t = 0.0
tMax = 2./nu
Dt = tMax/100.

i, j, k, l, n = indices(5)
f = Constant((0., 0., 0.)) #N/kg
r = Constant(0.0)
Tref = 300. #in K
Tamb = Tref
eps_0 = 8.85E-12 #in A s/(V m)
mu_0 = 12.6E-7 #in V s/(A m)
null=1E-20 #for numerical reasons it is not zero
h = 10. #in J / (s m^2 K)

# brass
E_b = 105E9 #in Pa
G_b = 36E9 #in Pa
la_b = (E_b-2.*G_b)*G_b/(3.*G_b-E_b)
mu_b = G_b
C_voigt_b = numpy.array([ \
[la_b+2.*mu_b, la_b, la_b, 0, 0, 0],\
[la_b, la_b+2.*mu_b, la_b, 0, 0, 0],\
[la_b, la_b, la_b+2.*mu_b, 0, 0, 0],\
[0, 0, 0, mu_b, 0, 0],\
[0, 0, 0, 0, mu_b, 0],\
[0, 0, 0, 0, 0, mu_b]  ])
alpha_b = 19E-6 #in 1/K
varsigma_b = 0.6 #in S/m or in 1/(Ohm m)
kappa_b = 109. #in W/(K m)
c_b = 380. #in J/(kg K)
rho_0_b = 8400. #in kg/m3

# PZT-5H material
S11_p = 15.6E-12 #1/Pa
S33_p = 20E-12 #1/Pa
nu_p = 0.31
S_voigt_p = numpy.array([ \
[S11_p, -nu_p*S11_p, -nu_p*S11_p, 0, 0, 0],\
[-nu_p*S11_p, S11_p, -nu_p*S11_p, 0, 0, 0],\
[-nu_p*S11_p, -nu_p*S11_p, S33_p, 0, 0, 0],\
[0, 0, 0, (1.+nu_p)*S11_p, 0, 0],\
[0, 0, 0, 0, (1.+nu_p)*S11_p, 0],\
[0, 0, 0, 0, 0, (1.+nu_p)*S11_p]  ])
C_voigt_p = numpy.linalg.inv(S_voigt_p)
dtilde_p_31 = -265E-12 #in m/V
dtilde_p_33 = 585E-12 #in m/V
dtilde_p_15 = 730E-12 #in m/V
eps_rel_el_p_11 = 3130.
eps_rel_el_p_33 = 3400.
alpha_p_11 = 6E-6 #in 1/K
alpha_p_33 =-4E-6 #in 1/K
varsigma_p = null
kappa_p = 1.1 #in W/(K m)
c_p = 350. #in J/(kg K)
rho_0_p = 7500. #in kg/m3

C11 = material_coefficient(mesh, cells_mat, [C_voigt_b[0,0], C_voigt_p[0,0]], element=coeff_element)
C12 = material_coefficient(mesh, cells_mat, [C_voigt_b[0,1], C_voigt_p[0,1]], element=coeff_element)
C13 = material_coefficient(mesh, cells_mat, [C_voigt_b[0,2], C_voigt_p[0,2]], element=coeff_element)
C22 = material_coefficient(mesh, cells_mat, [C_voigt_b[1,1], C_voigt_p[1,1]], element=coeff_element)
C33 = material_coefficient(mesh, cells_mat, [C_voigt_b[2,2], C_voigt_p[2,2]], element=coeff_element)
C44 = material_coefficient(mesh, cells_mat, [C_voigt_b[3,3], C_voigt_p[3,3]], element=coeff_element)
C = VoigtToTensorRank4(A11=C11,A12=C12,A13=C13,A22=C22,A23=C13,A33=C33,A44=C44,A55=C44,A66=C44)

dtilde31 = material_coefficient(mesh, cells_mat, [0., dtilde_p_31], element=coeff_element)
dtilde33 = material_coefficient(mesh, cells_mat, [0., dtilde_p_33], element=coeff_element)
dtilde15 = material_coefficient(mesh, cells_mat, [0., dtilde_p_15], element=coeff_element)
dtilde = VoigtToTensorRank3(A31=dtilde31, A32=dtilde31, A33=dtilde33, A15=dtilde15, A24=dtilde15) 
Ttilde = as_tensor( dtilde[i,j,k]*C[n,l,j,k], (i,n,l))

eps_rel_el11 = material_coefficient(mesh, cells_mat, [1., eps_rel_el_p_11], element=coeff_element)
eps_rel_el33 = material_coefficient(mesh, cells_mat, [1., eps_rel_el_p_33], element=coeff_element)
eps_rel_el = ToTensorRank2(A11=eps_rel_el11, A22=eps_rel_el11, A33=eps_rel_el33)
chi_el = eps_rel_el - delta

alpha11 = material_coefficient(mesh, cells_mat, [alpha_b, alpha_p_11], element=coeff_element)
alpha33 = material_coefficient(mesh, cells_mat, [alpha_b, alpha_p_33], element=coeff_element)
alpha = ToTensorRank2(A11=alpha11, A22=alpha11, A33=alpha33)

varsigma = material_coefficient(mesh, cells_mat, [varsigma_b, varsigma_p], element=coeff_element)
kappa = material_coefficient(mesh, cells_mat, [kappa_b, kappa_p], element=coeff_element)
c = material_coefficient(mesh, cells_mat, [c_b, c_p], element=coeff_element)
rho_0 = material_coefficient(mesh, cells_mat, [rho_0_b, rho_0_p], element=coeff_element)
v_0 = 1./rho_0


actuator = Expression("100.0*sin(2.0*pi*freq*t)", degree=1, freq=nu, t=0)
bc1 = DirichletBC(Space.sub(0), 0.0, facets, surfaces['clamp'])
bc2 = DirichletBC(Space.sub(0), actuator, facets, surfaces['loading'])
bc3 = DirichletBC(Space.sub(2), Constant((0.0, 0.0, 0.0)), facets, surfaces['clamp'])
bc = [bc1,bc2,bc3]

dunkn = TrialFunction(Space)
test = TestFunction(Space)
unkn = Function(Space)
unkn0 = Function(Space)
unkn00 = Function(Space)

#initial values for phi, A, u, T
unkn_init = Expression(('0.', '0.','0.','0.', '0.','0.','0.', 'T_r'), degree=1, T_r=Tref)
unkn00 = interpolate(unkn_init,Space)
unkn0.assign(unkn00)
unkn.assign(unkn0)

del_phi,del_A,del_u,del_T = split(test)
phi,A,u,T = split(unkn)
phi0,A0,u0,T0 = split(unkn0)
phi00,A00,u00,T00 = split(unkn00)

eps = sym(grad(u))
eps0 = sym(grad(u0))
E = -grad(phi) - (A-A0)/Dt
E0 = -grad(phi0) - (A0-A00)/Dt
B = as_tensor(epsilon[i,j,k]*A[k].dx(j) , (i,))
B0 = as_tensor(epsilon[i,j,k]*A0[k].dx(j) , (i,))
EE = as_tensor(E[i] + epsilon[i,j,k]*(u-u0)[j]/Dt*B[k], (i,))

#constitutive equations
D = eps_0*E
D0 = eps_0*E0
H = 1./mu_0*B
m = as_tensor( -1./2.*delta[i,j]*(H[k]*B[k]+D[k]*E[k]) + H[i]*B[j] + D[j]*E[i] , (j,i))
P = as_tensor(-Ttilde[i,j,k]*alpha[j,k]*(T-Tref) + Ttilde[i,j,k]*eps[j,k] + eps_0*chi_el[i,j]*E[j] , (i,))
P0 = as_tensor(-Ttilde[i,j,k]*alpha[j,k]*(T0-Tref) + Ttilde[i,j,k]*eps0[j,k] + eps_0*chi_el[i,j]*E0[j] , (i,))
mD = D + P
mD0 = D0 + P0
JJ_fr = varsigma*EE
J_fr = as_tensor( JJ_fr[i] + mD[j].dx(j)*(u-u0)[i]/Dt , (i,))
sigmaBar = as_tensor( -C[j,i,k,l]*alpha[k,l]*(T-Tref) , (j,i))
tau = as_tensor(m[j,i] + P[j]*E[i] + C[j,i,k,l]*eps[k,l] - Ttilde[k,j,i]*E[k] , (j,i))
eta = as_tensor( c*ln(T/Tref) + v_0*C[i,j,k,l]*alpha[k,l]*eps[i,j] - v_0*Ttilde[i,j,k]*alpha[j,k]*E[i] , ())
eta0 = as_tensor( c*ln(T0/Tref) + v_0*C[i,j,k,l]*alpha[k,l]*eps0[i,j] - v_0*Ttilde[i,j,k]*alpha[j,k]*E0[i] , ())
q = as_tensor(-kappa*T.dx(i), (i,))
Phi = q/T
Sigma = as_tensor(-q[i]/T/T*T.dx(i) + 1./T*EE[i]*JJ_fr[i], ())

qHat = h*(T-Tamb)
PhiHat = qHat/T

surfaces = {'singularity':1,'boundary_air':2,'clamp':5,'loading':6}

oldsurfaces = {'singularity':1,'boundary_air':2,'clamp':3,'loading':4}


#weak forms
F_phi = ( -(mD-mD0)[i]*del_phi.dx(i) - Dt*J_fr[i]*del_phi.dx(i) )*dV + N('+')[i]*Dt*(J_fr('+') - J_fr('-'))[i]*del_phi('+')*dI(surfaces['singularity'])

F_A = (eps_0*(A-2.*A0+A00)[i]/Dt/Dt*del_A[i] + 1./mu_0*A[i].dx(j)*del_A[i].dx(j) -J_fr[i]*del_A[i] - (P-P0)[i]/Dt*del_A[i] ) * dV

F_u = (rho_0*(u-2.*u0+u00)[i]/Dt/Dt*del_u[i] - sigmaBar[j,i].dx(j)*del_u[i] + tau[j,i]*del_u[i].dx(j) - rho_0*f[i]*del_u[i] )*dV + N[j]*sigmaBar[j,i]*del_u[i]*(dA(surfaces['boundary_air'])+dA(surfaces['loading']))

F_T = ( rho_0*(eta-eta0)*del_T - Dt*Phi[i]*del_T.dx(i) -Dt*rho_0*r/T*del_T - Dt*Sigma*del_T )*(dV(1)+dV(2)) + Dt*PhiHat*del_T*(dA(surfaces['boundary_air'])+dA(surfaces['loading'])+dA(surfaces['clamp']))

Form = F_phi + F_A + F_u + F_T
Gain = derivative(Form, unkn, dunkn)

pwd = 'results/CR19/'
file_phi = File(pwd + 'phi.pvd')
file_A = File(pwd + 'A.pvd')
file_u = File(pwd + 'u.pvd')
file_T = File(pwd + 'T.pvd')

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pylab
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

pylab.rc('text', usetex=True )
pylab.rc('font', family='serif', serif='cm', size=30 )
pylab.rc('legend', fontsize=30)
pylab.rc(('xtick.major','ytick.major'), pad=15)
c1, c2 = '#990000', '#0033FF'
fig, ax1 = pylab.subplots(figsize = (14, 8))
pylab.tight_layout()
pylab.subplots_adjust(top=0.95)  
pylab.subplots_adjust(bottom=0.15)
pylab.subplots_adjust(left=0.1)
pylab.subplots_adjust(right=0.9)
ax1.ticklabel_format(style='sci',scilimits=(-3,+3),axis='y')
ax1.grid(True, axis='x')
ax1.set_xlabel(r'$t$ in s')
ax1.set_ylabel(r'$\phi$ in V', color=c1)
ax1.tick_params(axis='y', colors=c1)
ax1.grid(True, axis='y', color=c1)
ax2 = ax1.twinx()
ax2.ticklabel_format(style='sci',scilimits=(-3,+3),axis='y')
ax2.set_ylabel(r'$u_3$ in mm', color=c2)
ax2.tick_params(axis='y', colors=c2)
ax2.grid(True, axis='y', color=c2)

time_plot, phi_plot, u_plot = [0],[0],[0]
while t < tMax:
	t += Dt
	actuator.t = t
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('time: %.2e s computed in %.0f h %.0f min %.0f s' % (t, e_h, e_m, e_s )  )

	solve(Form == 0, unkn, bc, J=Gain, \
		solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-5} }, \
		form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2}  )


	time_plot.append(t)
	phi_plot.append(unkn.split()[0](0.,0.,-0.0001))
	u_plot.append(unkn.split()[2](0.,0.,0.0005)[2]*1000.)

	ax1.plot(time_plot, phi_plot, color=c1, linewidth=3, linestyle='-', marker='o', markersize=8)
	ax2.plot(time_plot, u_plot, color=c2, linewidth=3, linestyle='--', marker='o', markersize=8)

	pylab.savefig(pwd + 'CompReal19_input_output.pdf')
	file_phi << (unkn.split()[0], t)
	file_A << (unkn.split()[1], t)
	file_u << (unkn.split()[2], t)
	file_T << (unkn.split()[3], t)

	unkn00.assign(unkn0)
	unkn0.assign(unkn)
