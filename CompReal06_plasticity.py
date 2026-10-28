"""Computational reality 06, plasticity"""
__author__ = "B. Emek Abali"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html


import time as comp_time
start_time = comp_time.time()

from dolfin import *
from ufl_legacy import indices
import numpy
set_log_level(50)

xlength =100. #[mm]
ylength =10. #[mm]
zlength =10. #[mm]
mesh = BoxMesh(Point(0, 0, 0), Point(xlength, ylength, zlength), 10, 3, 3)
Dim = mesh.geometric_dimension()
Coeff = FunctionSpace(mesh,'CG',1)
Space = VectorFunctionSpace(mesh,'CG',1)
Tensor = TensorFunctionSpace(mesh,'CG',1)
delu = TestFunction(Space)
du = TrialFunction(Space)
u = Function(Space)
u0 = Function(Space)
u00 = Function(Space)

cells = MeshFunction('size_t', mesh, Dim)
facets = MeshFunction('size_t', mesh, Dim-1)
dA = Measure('ds', domain=mesh, subdomain_data=facets)
dV = Measure('dx', domain=mesh, subdomain_data=cells)

print('initializing,time')
t=0.0
t_end = 10.0
dt = 0.05

#Defining boundary conditions
left = CompiledSubDomain('near(x[0],0) && on_boundary')
right = CompiledSubDomain('near(x[0],l) && on_boundary',l=xlength)

# a cyclic displacement on the right end
displ = Expression(('0.5*sin(2.*pi*f*time)','0.0','0.0'),degree=4,f=0.1,time=0) 
bc1 = DirichletBC(Space,displ,right)
bc2 = DirichletBC(Space,(0.0,0.0,0.0),left)
bc = [bc1,bc2]

#Setting up the material parameters in tonne, seconds, Newton, milimeter
rho0 = 8.3E-9 #tonne/mm^3
nu, E = 0.3, 200000.0 # [-], [MPa]
h=0.01*E
lambada=E*nu/(1.0+nu)/(1.0-2.0*nu)
mu=E/(2.0+2.0*nu)

sigmaY=Constant(300.0) # [MPa]
# index notation
i,j,k,l,m,n,o,p = indices(8)
delta= Identity(3)
eps= as_tensor(1.0/2.0*(u[i].dx(j)+u[j].dx(i)),(i,j))
eps0= as_tensor(1.0/2.0*(u0[i].dx(j)+u0[j].dx(i)),(i,j))
epsDot=as_tensor((eps[i,j]-eps0[i,j])/dt,(i,j))
gamma=Function(Coeff)

C = as_tensor(lambada*delta[i,j]*delta[k,l]\
+mu*delta[i,k]*delta[j,l]+mu*delta[i,l]*delta[j,k],(i,j,k,l))

flow_cond_0 = Function(Coeff)
direction_cond_0 = Function(Coeff)
sigma0 = Function(Tensor)
dev_sigma0 = as_tensor(sigma0[i,j]-1./3.*sigma0[k,k]*delta[i,j],(i,j))
beta0=Function(Tensor)
stress_diff = dev_sigma0-beta0

sigmaDot = as_tensor( (C[i,j,m,n]-gamma*C[i,j,k,l]*(dev_sigma0[k,l]\
-beta0[k,l])*(dev_sigma0[o,p]-beta0[o,p])*C[o,p,m,n]/(4./9.*sigmaY**2*h\
+(dev_sigma0[i,j]-beta0[i,j])*C[i,j,k,l]*(dev_sigma0[k,l]\
-beta0[k,l]) ) )*epsDot[m,n], (i,j) )

sigma = as_tensor(sigma0[i,j]+dt*sigmaDot[i,j],(i,j))
dev_sigma = as_tensor(sigma[i,j]-1./3.*sigma[k,k]*delta[i,j],(i,j))

betaDot = as_tensor( gamma*(dev_sigma0[k,l]-beta0[k,l])*sigmaDot[k,l]\
/(2.0/3.0*sigmaY**2)*(dev_sigma0[i,j]-beta0[i,j]),(i,j))

beta=as_tensor(beta0[i,j]+dt*betaDot[i,j],(i,j))

f= Constant((0.,0.,0.))

Form = (rho0*(u[i]-2.*u0[i]+u00[i])/(dt*dt)*delu[i] \
	+ sigma[j,i]*delu[i].dx(j) \
	- rho0*f[i]*delu[i] )*dV
Gain = derivative(Form,u,du)

#Plotting stress vs. strain curves
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pylab
pylab.rc('text', usetex=True )
pylab.rc('font', family='serif', serif='cm', size=30 )
pylab.rc('legend', fontsize=30)
pylab.rc(('xtick.major','ytick.major'), pad=15)

#pylab.ion()
fig = pylab.figure(1, figsize=(12,8))
fig.clf()
pylab.subplots_adjust(bottom=0.18)
pylab.subplots_adjust(left=0.16)
pylab.xlabel(r'strain $\varepsilon_{xx}$')
pylab.ylabel(r'stress $\sigma_{xx}$ in MPa')
pylab.grid(True)

stress_plot = []
strain_plot = []
temp_array= []
time= []
uinit = Expression(('0.0','0.0','0.0'),degree=1)
u0.interpolate(uinit)
u00.assign(u0)
file_u = File ('results/CR06/displacements.pvd')

s_type = 'mumps'
f_param = {"cpp_optimize": True, "representation":"uflacs",  "quadrature_degree": 2}

while t <= t_end :
	displ.time = t
	
	solve(Form == 0, u, bc, J=Gain, \
		solver_parameters={"newton_solver":{"linear_solver": "mumps", "relative_tolerance": 1e-10} }, \
		form_compiler_parameters=f_param  )

	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('time: %.2f s computed in %.0f h %.0f min %.0f s ' % (t, e_h, e_m, e_s )  )

	sigma_ = project(sigma,Tensor, solver_type=s_type, form_compiler_parameters=f_param )
	strain_plot.append(u(xlength,ylength/2.,zlength/2.)[0]/xlength)
	stress_plot.append(sigma_(xlength/2.,ylength/2.,zlength/2.)[0])
	file_u << (u,t)
	sigma0.assign(sigma_)
	beta_ = project(beta,Tensor, solver_type=s_type, form_compiler_parameters=f_param )
	beta0.assign(beta_)

	flow_cond = as_tensor(1./2.*stress_diff[i,j]*stress_diff[i,j] - 1./3.*sigmaY**2, ())
	direction_cond = as_tensor( stress_diff[i,j]*epsDot[i,j], ())

	assign(flow_cond_0, project(flow_cond, Coeff, solver_type=s_type, form_compiler_parameters=f_param))    
	assign(direction_cond_0, project(direction_cond, Coeff, solver_type=s_type, form_compiler_parameters=f_param))
	gamma_= Expression ("flow_cond >= 0 && direction > 0 ? 1: 0", flow_cond = flow_cond_0, direction = direction_cond_0, degree = 3)
	assign(gamma, project(gamma_, Coeff, solver_type=s_type, form_compiler_parameters=f_param)) 


	u00.assign(u0)
	u0.assign(u)
	t = t + dt

	pylab.plot(strain_plot,stress_plot,color='red',marker='o',markersize=5)
	#pylab.draw()

	pylab.savefig('results/CR06_plast_tensile.pdf')

