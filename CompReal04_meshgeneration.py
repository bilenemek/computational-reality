"""Computational reality 04, mesh generation for a nanoindentation simluation."""
__author__ = "B. Emek Abali"
__license__  = "GNU LGPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

from dolfin import *
xL=20.0 #micrometer
yL=20.0
zL=20.0
# origin is on the middle of top surface
mesh = BoxMesh(Point(-xL/2, -yL/2, -zL), Point(xL/2, yL/2, 0), 25,25,25)
center=Point(0,0,0)
def refine_interval(radius,mesh):
	markers = MeshFunction("bool", mesh, 3)
	markers.set_all(False)
	# Mark cells for refinement
	for cell in cells(mesh):
		if cell.midpoint().distance(center) < radius:
			markers[cell.index()] = True
	# Refine mesh
	mesh = refine(mesh, markers)
	print('DOFs: ', mesh.num_vertices()*3)
	return mesh

num_refinements = 1
refine_radius1 = 3.0
refine_radius2 = 1.0
refine_radius3 = 0.3
refine_radius4 = 0.1
for i in range(num_refinements):
	mesh=refine_interval(refine_radius1,mesh)
	mesh=refine_interval(refine_radius2,mesh)
	mesh=refine_interval(refine_radius3,mesh)
	mesh=refine_interval(refine_radius4,mesh)

pwd='results/'
meshfile = File(pwd+'nanoindent_mesh.xml')
meshfile << mesh

