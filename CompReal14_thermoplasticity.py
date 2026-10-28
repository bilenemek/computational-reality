"""Computational reality 14, thermoplasticity"""
__author__ = "B. Emek Abali"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

import time as comp_time
start_time = comp_time.time()

from dolfin import *
from ufl_legacy import indices
import numpy
set_log_level(50)

#units: mm, 1000 kg=ton, s, MPa, mJ, K
Tref = 300. #in K
Tamb=Tref
# Material data of P265GH (St 45.8) from VDI Waermeatlas, at 293.15 K
rho = 7.85E-9 #in tonne / mm^3
kappa = 57.0 #in mJ / (s mm K)
capacity = 430.0E6 #in mJ / (ton K)
alpha = 12.2E-6 #in 1/K at 373.15 K
EModul = 211.0E+3 #in MPa
nu = 0.28
H =  2600. #MPa    
h = 10.E-3 #in mJ / (s mm^2 K)
sigmaY = Constant(250.0) #MPa

tMax = 10.0
Dt = 0.2
t = 0.0

xMin, xMax, xElements = 0.0, 100.0, 20
yMin, yMax, yElements = -5., +5., 2
zMin, zMax, zElements = -5., +5., 2
mesh = BoxMesh(Point(xMin,yMin,zMin), Point(xMax,yMax,zMax), xElements,yElements,zElements)
Dim = mesh.geometric_dimension()
N = FacetNormal(mesh)
length = abs(xMax-xMin)

Coeff = FunctionSpace(mesh,'CG',1)
Tensor = TensorFunctionSpace(mesh, 'CG', 1)
scalar = FiniteElement("CG", mesh.ufl_cell(), 1) 
vector = VectorElement("CG", mesh.ufl_cell(), 1) 
# T, u
Space = FunctionSpace( mesh, MixedElement([scalar, vector]) )

cells = MeshFunction('size_t',mesh,Dim)
facets = MeshFunction('size_t',mesh,Dim-1)
dA = Measure('ds', domain=mesh, subdomain_data=facets)
dV = Measure('dx', domain=mesh, subdomain_data=cells)

left = CompiledSubDomain('near(x[0],l) && on_boundary',l=xMin)
right = CompiledSubDomain('near(x[0],l) && on_boundary',l=xMax)
back = CompiledSubDomain('near(x[1],l) && on_boundary',l=yMin)
front = CompiledSubDomain('near(x[1],l) && on_boundary',l=yMax)
bottom = CompiledSubDomain('near(x[2],l) && on_boundary',l=zMin)
top = CompiledSubDomain('near(x[2],l) && on_boundary',l=zMax)

facets.set_all(0)
displ = Expression(('0.5*sin(2.*pi*f*time)','0.0','0.0'),degree=4,f=0.1,time=0)
bc1 = DirichletBC(Space.sub(1),displ,right)
bc2 = DirichletBC(Space.sub(1),Constant((0.0,0.0,0.0)),left)

bc = [bc1,bc2]

dunkn = TrialFunction(Space)
test = TestFunction(Space)
delT, delu = split(test)

unkn = Function(Space)
unkn0 = Function(Space)
unkn00 = Function(Space)

unkn_init = Expression(('Tini','0','0','0'),degree=1,Tini=Tref)
unkn = interpolate(unkn_init,Space)
unkn0.assign(unkn)
unkn00.assign(unkn0)

T, u = split(unkn)
T0, u0 = split(unkn0)
T00, u00 = split(unkn0)

i, j, k, l, m, n, o, p, r, s = indices(10)
delta = Identity(3)
lambada = EModul * nu / (1.+nu) / (1.-2.*nu)
mu = 0.5 * EModul / (1.+nu)
C = as_tensor(lambada*delta[i,j]*delta[k,l]+mu*delta[i,k]*delta[j,l]+mu*delta[i,l]*delta[j,k], (i,j,k,l))
alp = alpha*delta

flow_cond_0 = Function(Coeff)
direction_cond_0 = Function(Coeff)
peps0 = Function(Tensor)
sigma0= Function(Tensor)
dev_sigma0 = as_tensor(sigma0[i,j]-1./3.*sigma0[k,k]*delta[i,j], (i,j))
beta0 = Function(Tensor)
stress_diff = dev_sigma0-beta0


eps = sym(grad(u))
eps0 = sym(grad(u0))
Deps = (eps-eps0)/Dt

teps = alp*(T-Tref)
Dteps = alp*(T-T0)/Dt

gamma = Function(Coeff)
Dpeps = as_tensor(gamma*(dev_sigma0-beta0)[i,j]*C[i,j,k,l]*(Deps-Dteps)[k,l]/(4./9.*H*sigmaY**2+(dev_sigma0-beta0)[m,n]*C[m,n,o,p]*(dev_sigma0-beta0)[o,p])*(dev_sigma0-beta0)[r,s], (r,s))

Dsigma = as_tensor(C[i,j,k,l]*(Deps-Dpeps-Dteps)[k,l], (i,j))
Dbeta = as_tensor( gamma*(dev_sigma0-beta0)[k,l]*Dsigma[k,l]/(2./3.*sigmaY**2)*(dev_sigma0-beta0)[i,j], (i,j))

sigma = sigma0 + Dt*Dsigma
beta = beta0 + Dt*Dbeta
peps = peps0 + Dt*Dpeps

dev_sigma = as_tensor(sigma[i,j]-1./3.*sigma[k,k]*delta[i,j], (i,j))
q = as_tensor(-kappa*T.dx(i), (i,))

f = Constant((0.0,0.0,0.0))
R = Constant(0.0)
qHat = h*(T-Tamb)

F_u = (rho*(u-2.*u0+u00)[i]/Dt/Dt*delu[i] + sigma[j,i]*delu[i].dx(j) - rho*f[i]*delu[i] )*dV
F_T = (rho*capacity/T*(T-T0)*delT+ Dt*alp[i,j]*C[i,j,k,l]*(Deps-Dpeps-Dteps)[k,l]*delT - Dt*q[i]*(delT/T).dx(i) - Dt*rho*R/T*delT - Dt*sigma0[i,j]*Dpeps[j,i]*delT/T )*dV + Dt*qHat*delT/T*dA

Form = F_u + F_T
Gain = derivative(Form, unkn, dunkn)

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pylab
pylab.rc('text', usetex=True )
pylab.rc('font', family='serif', serif='cm', size=20 )
pylab.rc('legend', fontsize=20)
pylab.rc(('xtick.major','ytick.major'), pad=5)
pylab.subplots_adjust(top=0.90)
pylab.subplots_adjust(bottom=0.17)
pylab.subplots_adjust(left=0.20)
pylab.subplots_adjust(right=0.75)

fig = pylab.figure(1, figsize=(14,10))
ax1 = fig.add_subplot(111)
ax1.grid(True, axis='x')
ax1.set_xlabel(r'$u_1/l_0$ in $\%$')
ax1.set_ylabel(r'$\sigma_{11}$ in MPa', color='r')
ax1.tick_params(axis='y', colors='r')
ax1.grid(True, axis='y', color='r')
ax2 = ax1.twinx()
ax2.set_ylabel(r'$(T-T_\mathrm{ref})$ in K', color='b')
ax2.tick_params(axis='y', colors='b')
ax2.grid(True, axis='y', color='b')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))

pwd='results/CR14/'
file_u = File(pwd+'displ.pvd')
file_T = File(pwd+'temp.pvd')
strain,stress,temp =[],[],[]

s_type = 'mumps'
f_param = {"cpp_optimize": True, "representation":"uflacs",  "quadrature_degree": 2}

while t < tMax:
	displ.time = t
	solve(Form == 0, unkn, bc, J=Gain, \
		solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-5} }, \
		form_compiler_parameters=f_param  )

	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('time: %.2e s computed in %.0f h %.0f min %.0f s ' % (t, e_h, e_m, e_s )  )

	file_T << (unkn.split()[0], t)
	file_u << (unkn.split()[1], t)

	sigma_ = project(sigma, Tensor, solver_type="mumps", form_compiler_parameters=f_param )
	sigma0.assign(sigma_)
	beta_ = project(beta, Tensor, solver_type="mumps", form_compiler_parameters=f_param )
	beta0.assign(beta_)

	flow_cond = as_tensor(1./2.*stress_diff[i,j]*stress_diff[i,j] - 1./3.*sigmaY**2, ())
	direction_cond = as_tensor( stress_diff[i,j]*Deps[i,j], ())

	assign(flow_cond_0, project(flow_cond, Coeff, solver_type=s_type, form_compiler_parameters=f_param))    
	assign(direction_cond_0, project(direction_cond, Coeff, solver_type=s_type, form_compiler_parameters=f_param))
	gamma_= Expression ("flow_cond >= 0 && direction > 0 ? 1: 0", flow_cond = flow_cond_0, direction = direction_cond_0, degree = 3)
	assign(gamma, project(gamma_, Coeff, solver_type=s_type, form_compiler_parameters=f_param)) 

	peps_ = project(peps, Tensor, solver_type="mumps", form_compiler_parameters=f_param )
	peps0.assign(peps_)

	unkn00.assign(unkn0)
	unkn0.assign(unkn)

	strain.append(unkn0.split()[1](xMax,0.,0.)[0]/length*100.)
	stress.append(sigma0(xMax/2.,0.,0.)[0])
	temp.append(unkn0.split()[0](xMax/2.,0.,0.)-Tref)
	ax1.plot(strain,stress, 'o-', color='r')
	ticks = numpy.linspace(numpy.array(strain).min(), numpy.array(strain).max(), 4)
	ax1.set_xticks(ticks)
	ax1.set_xticklabels(['%1.2f' % i_ticks for i_ticks in ticks] )
	ax2.plot(strain,temp, 'd-', color='b')
	fig.savefig(pwd+'CompReal14_tensiletest.pdf')

	t += Dt
