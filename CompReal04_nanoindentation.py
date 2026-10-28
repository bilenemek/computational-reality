"""Computational reality 04, simulation of an nanoindentation modeled with a Zener model."""
__author__ = "B. Emek Abali"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

from dolfin import *
from ufl_legacy import indices

#set_log_level(50)
mesh = Mesh('results/nanoindent_mesh.xml')
mesh.order()
D = mesh.topology().dim()
xL=20.0 #micrometer
yL=20.0
zL=20.0

V = VectorFunctionSpace(mesh, 'CG', 1)
T = TensorFunctionSpace(mesh, 'CG', 1)

left = CompiledSubDomain('near(x[0],l) && on_boundary',l=-xL/2)
right = CompiledSubDomain('near(x[0],l) && on_boundary',l=xL/2)
back = CompiledSubDomain('near(x[1],l) && on_boundary',l=-yL/2)
front= CompiledSubDomain('near(x[1],l) && on_boundary',l=yL/2)
bottom = CompiledSubDomain('near(x[2],l) && on_boundary',l=-zL)

u_max = 0.1
A = 24.5*u_max**2
radius = sqrt(A/pi)

top = CompiledSubDomain('near(x[2], 0) && x[0]*x[0]+x[1]*x[1] <= rad*rad',rad=radius)

cells = MeshFunction('size_t', mesh, 3)
facets = MeshFunction('size_t', mesh, 2)
dA = Measure('ds', domain=mesh, subdomain_data=facets)
dV = Measure('dx', domain=mesh, subdomain_data=cells)

facets.set_all(0)
top.mark(facets, 1)
nanoindent = Expression(('0.0','0.0','-Amp'),degree=2,Amp=1.)

bc = [DirichletBC(V, (0.,0.,0.), bottom)]
f_gr = Constant((0.,0.,0.))

du = TrialFunction(V)
delu = TestFunction(V)
u00 = Function(V)
u0 = Function(V)
u = Function(V)
S=Function(T)
S0=Function(T)

print ('initializing, time in ms')
t = 0.0
tend = 3000.0
dt = 50.

init = Expression(('0','0','0'),degree=1)
u.interpolate(init)
u0.assign(u)
u00.assign(u0)

print ('initializing, space')
rho0 = 9000.0E-15 #kg/mikrometer^3
lambada = 90.0 # mN/mikrometer^2 (=GPa)
E1, E2 = 200.0, 200.0 # mN/mikrometer^2 (=GPa)
mu = 2.0E5 #mN ms / mikrometer^2 (=N s/mm^2)

i, j, k, r = indices(4)
delta = Identity(3)

F = as_tensor(delta[i,j]+u[i].dx(j), [i,j])

eps= as_tensor(1.0/2.0*(u[i].dx(k)+u[k].dx(i)), [i,k])
eps0= as_tensor(1.0/2.0*(u0[i].dx(k)+u0[k].dx(i)), [i,k])

eps_dev= as_tensor(eps[i,j]-eps[k,k]*1.0/3.0*delta[i,j], [i,j])
eps0_dev= as_tensor(eps0[i,j]-eps0[k,k]*1.0/3.0*delta[i,j], [i,j])

devS0= as_tensor(S0[i,j]-S0[k,k]*1.0/3.0*delta[i,j], [i,j])

S = as_tensor(lambada*eps[j,j]*delta[i,k] + mu/(E2*dt+mu)*devS0[i,k] \
+ E1*E2*dt/(E2*dt+mu)*eps_dev[i,k] + mu*(E1+E2)/(E2*dt+mu)*(eps_dev[i,k]-eps0_dev[i,k]), [i,k])

N = FacetNormal(mesh)

Form = (rho0/dt/dt*(u-2.*u0+u00)[i]*delu[i] + F[i,k]*S[r,k]*delu[i].dx(r) - \
rho0*f_gr[i]*delu[i])*dV - nanoindent[i]*delu[i]*dA(1)

nz = as_tensor([0.0,0.0,1.0])
forceZ = as_tensor(F[i,k]*S[r,k]*nz[r],[i,])

Gain = derivative(Form, u, du)

pwd='results/CR04/'
file_u = File(pwd+'displacement.pvd')

time_values=[0.0]
u_max_values=[0.0]
forceZ_values=[0.0]
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
pylab.xlabel(r'indentation depth $u_{\mathrm{max}}$ in $\mu$m')
pylab.ylabel(r'force $f(t)$ in mN')
pylab.grid(True)


s_type = 'gmres'
f_param = {"cpp_optimize": True, "representation":"uflacs",  "quadrature_degree": 2}

while t<tend:
	t += dt
	print ('time: ', t, ' indent: ', u_max)
	if t<1000.: nanoindent.Amp = 0.002*t/A
	if t>=1000. and t<=1500.: nanoindent.Amp = 2./A
	if t>1500.: nanoindent.Amp = (2.-0.002*(t-1500.))/A
	if t>=2500.:nanoindent.Amp = 0.

	solve(Form == 0, u, bc, J=Gain, \
		solver_parameters={"newton_solver":{"linear_solver": "cg", "preconditioner": "hypre_amg", "relative_tolerance": 1E-2, "absolute_tolerance": 1E-5, "maximum_iterations": 30} }, \
		form_compiler_parameters=f_param  )

	file_u << (u,t)
	time_values.append(t)
	u_max = abs(u((0,0,0))[2])
	facets.set_all(0)
	top.mark(facets, 1)
	#if u_max<0.02: u_max=0.02
	u_max_values.append(u_max)
	fZ = project(forceZ,V)
	fZvalue = abs(fZ((0,0,0))[2])*A
	forceZ_values.append(fZvalue)
	#print (time_values,u_max_values,forceZ_values)
	S0.assign(project(S,T,solver_type=s_type, form_compiler_parameters=f_param))
	u00.assign(u0)
	u0.assign(u)
	#file_list.write(" time=\"%s\" u(0,0,0)=\"%s\" force(0,0,0)=\"%s\" \n" % (t, u_max, fZvalue))
	pylab.plot(u_max_values, forceZ_values, 'ro-')
	pylab.savefig(pwd+'Nanoindentation.pdf')

#file_list.close()

