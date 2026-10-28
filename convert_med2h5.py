"""
Computational reality, converting med from Salome to h5 file for FEniCS

by B. E. Abali ...
"""
__author__ = "B. Emek Abali"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

# -*- coding: utf-8 -*-
import os
import getopt
import subprocess
import sys
import re
import warnings
import numpy
from dolfin import *

filename=str(sys.argv[1])
gmsh_path = './gmsh-4.7.1-Linux64/bin/gmsh'

keepmsh = False
if len(sys.argv) > 2:
    if str(sys.argv[2]) == '-k':
        keepmsh = True
    elif str(sys.argv[2]) == '-h':
        print('-h: show this help')
        print('-k: keep .msh files after conversion')
        exit()

# diese Klasse stammt aus dem Strukturpfad von dolfin-convert
class xml_writer():
    # Write mesh header
    def write_header_mesh(ofile, cell_type, dim):
        ofile.write("""\
<?xml version=\"1.0\" encoding=\"UTF-8\"?>

<dolfin xmlns:dolfin=\"http://www.fenicsproject.org\">
  <mesh celltype="%s" dim="%d">
""" % (cell_type, dim))

    # Write graph header
    def write_header_graph(ofile, graph_type):
        ofile.write("""\
<?xml version=\"1.0\" encoding=\"UTF-8\"?>

<dolfin xmlns:dolfin=\"http://www.fenicsproject.org/\">
  <graph type="%s">
""" % (graph_type))

    # Write mesh footer
    def write_footer_mesh(ofile):
        ofile.write("""\
  </mesh>
</dolfin>
""")

    # Write graph footer
    def write_footer_graph(ofile):
        ofile.write("""\
  </graph>
</dolfin>
""")

    def write_header_vertices(ofile, num_vertices):
        "Write vertices header"
        print("Expecting %d vertices" % num_vertices)
        ofile.write("    <vertices size=\"%d\">\n" % num_vertices)

    def write_footer_vertices(ofile):
        "Write vertices footer"
        ofile.write("    </vertices>\n")
        print("Found all vertices")

    def write_header_edges(ofile, num_edges):
        "Write edges header"
        print("Expecting %d edges" % num_edges)
        ofile.write("    <edges size=\"%d\">\n" % num_edges)

    def write_footer_edges(ofile):
        "Write edges footer"
        ofile.write("    </edges>\n")
        print("Found all edges")

    def write_vertex(ofile, vertex, *args):
        "Write vertex"
        coords = " ".join(['%s="%.16e"' % (comp, num) for (comp, num) in zip(["x","y","z"], args)])
        ofile.write('      <vertex index="%d" %s/>\n' % \
                    (vertex, coords))

    def write_graph_vertex(ofile, vertex, num_edges, weight = 1):
        "Write graph vertex"
        ofile.write("      <vertex index=\"%d\" num_edges=\"%d\" weight=\"%d\"/>\n" % \
            (vertex, num_edges, weight))

    def write_graph_edge(ofile, v1, v2, weight = 1):
        "Write graph edge"
        ofile.write("      <edge v1=\"%d\" v2=\"%d\" weight=\"%d\"/>\n" % \
            (v1, v2, weight))

    def write_header_cells(ofile, num_cells):
        "Write cells header"
        ofile.write("    <cells size=\"%d\">\n" % num_cells)
        print("Expecting %d cells" % num_cells)

    def write_footer_cells(ofile):
        "Write cells footer"
        ofile.write("    </cells>\n")
        print("Found all cells")

    def write_cell_interval(ofile, cell, n0, n1):
        "Write cell (interval)"
        ofile.write("      <interval index=\"%d\" v0=\"%d\" v1=\"%d\"/>\n" % \
            (cell, n0, n1))

    def write_cell_triangle(ofile, cell, n0, n1, n2):
        "Write cell (triangle)"
        ofile.write("      <triangle index=\"%d\" v0=\"%d\" v1=\"%d\" v2=\"%d\"/>\n" % \
            (cell, n0, n1, n2))

    def write_cell_tetrahedron(ofile, cell, n0, n1, n2, n3):
        "Write cell (tetrahedron)"
        ofile.write("      <tetrahedron index=\"%d\" v0=\"%d\" v1=\"%d\" v2=\"%d\" v3=\"%d\"/>\n" % \
            (cell, n0, n1, n2, n3))

    def write_header_domains(ofile):
        ofile.write("    <domains>\n")

    def write_footer_domains(ofile):
        ofile.write("    </domains>\n")

    def write_header_meshvaluecollection(ofile, name, dim, size, etype):
        ofile.write("    <mesh_value_collection name=\"%s\" type=\"%s\" dim=\"%d\" size=\"%d\">\n" % (name, etype, dim, size))

    def write_entity_meshvaluecollection(ofile, dim, index, value, local_entity=0):
        ofile.write("      <value cell_index=\"%d\" local_entity=\"%d\" value=\"%d\"/>\n" % (index, local_entity, value))

    def write_footer_meshvaluecollection(ofile):
        ofile.write("    </mesh_value_collection>\n")

    def write_header_meshfunction(ofile, dimensions, size):
        header = """<?xml version="1.0" encoding="UTF-8"?>
<dolfin xmlns:dolfin="http://fenicsproject.org">
  <mesh_function type="uint" dim="%d" size="%d">
""" % (dimensions, size)
        ofile.write(header)

    def write_header_meshfunction2(ofile):
        header = """<?xml version="1.0" encoding="UTF-8"?>
<dolfin xmlns:dolfin="http://fenicsproject.org">
  <mesh_function>
"""
        ofile.write(header)

    def write_entity_meshfunction(ofile, index, value):
        ofile.write("""    <entity index=\"%d\" value=\"%d\"/>
"""     % (index, value))

    def write_footer_meshfunction(ofile):
        ofile.write("""  </mesh_function>
</dolfin>""")


class ParseError(Exception):
    """ Error encountered in source file.
    """

class DataHandler(object):
    """ Baseclass for handlers of mesh data.

    The actual handling of mesh data encountered in the source file is
    delegated to a polymorfic object. Typically, the delegate will write the
    data to XML.
    @ivar _state: the state which the handler is in, one of State_*.
    @ivar _cell_type: cell type in mesh. One of CellType_*.
    @ivar _dim: mesh dimensions.
    """
    State_Invalid, State_Init, State_Vertices, State_Cells, \
          State_MeshFunction, State_MeshValueCollection = list(range(6))
    CellType_Tetrahedron, CellType_Triangle, CellType_Interval = list(range(3))

    def __init__(self):
        self._state = self.State_Invalid

    def set_mesh_type(self, cell_type, dim):
        assert self._state == self.State_Invalid
        self._state = self.State_Init
        if cell_type == "tetrahedron":
            self._cell_type = self.CellType_Tetrahedron
        elif cell_type == "triangle":
            self._cell_type = self.CellType_Triangle
        elif cell_type == "interval":
            self._cell_type = self.CellType_Interval
        self._dim = dim

    def start_vertices(self, num_vertices):
        assert self._state == self.State_Init
        self._state = self.State_Vertices

    def add_vertex(self, vertex, coords):
        assert self._state == self.State_Vertices

    def end_vertices(self):
        assert self._state == self.State_Vertices
        self._state = self.State_Init

    def start_cells(self, num_cells):
        assert self._state == self.State_Init
        self._state = self.State_Cells

    def add_cell(self, cell, nodes):
        assert self._state == self.State_Cells

    def end_cells(self):
        assert self._state == self.State_Cells
        self._state = self.State_Init

    def start_domains(self):
        assert self._state == self.State_Init

    def end_domains(self):
        self._state = self.State_Init

    def start_meshfunction(self, name, dim, size):
        assert self._state == self.State_Init
        self._state = self.State_MeshFunction

    def add_entity_meshfunction(self, index, value):
        assert self._state == self.State_MeshFunction

    def end_meshfunction(self):
        assert self._state == self.State_MeshFunction
        self._state = self.State_Init

    def start_mesh_value_collection(self, name, dim, size, etype):
        assert self._state == self.State_Init
        self._state = self.State_MeshValueCollection

    def add_entity_mesh_value_collection(self, dim, index, value, local_entity=0):
        assert self._state == self.State_MeshValueCollection

    def end_mesh_value_collection(self):
        assert self._state == self.State_MeshValueCollection
        self._state = self.State_Init

    def warn(self, msg):
        """ Issue warning during parse.
        """
        warnings.warn(msg)

    def error(self, msg):
        """ Raise error during parse.

        This method is expected to raise ParseError.
        """
        raise ParseError(msg)

    def close(self):
        self._state = self.State_Invalid

class XmlHandler(DataHandler):
    """ Data handler class which writes to Dolfin XML.
    """
    def __init__(self, ofilename):
        DataHandler.__init__(self)
        self._ofilename = ofilename
        self.__ofile = open(ofilename, "w")
        self.__ofile_meshfunc = None

    def ofile(self):
        return self.__ofile

    def set_mesh_type(self, cell_type, dim):
        DataHandler.set_mesh_type(self, cell_type, dim)
        xml_writer.write_header_mesh(self.__ofile, cell_type, dim)

    def start_vertices(self, num_vertices):
        DataHandler.start_vertices(self, num_vertices)
        xml_writer.write_header_vertices(self.__ofile, num_vertices)

    def add_vertex(self, vertex, coords):
        DataHandler.add_vertex(self, vertex, coords)
        xml_writer.write_vertex(self.__ofile, vertex, *coords)

    def end_vertices(self):
        DataHandler.end_vertices(self)
        xml_writer.write_footer_vertices(self.__ofile)

    def start_cells(self, num_cells):
        DataHandler.start_cells(self, num_cells)
        xml_writer.write_header_cells(self.__ofile, num_cells)

    def add_cell(self, cell, nodes):
        DataHandler.add_cell(self, cell, nodes)
        if self._cell_type == self.CellType_Tetrahedron:
            func = xml_writer.write_cell_tetrahedron
        elif self._cell_type == self.CellType_Triangle:
            func = xml_writer.write_cell_triangle
        elif self._cell_type == self.CellType_Interval:
            func = xml_writer.write_cell_interval

        func(self.__ofile, cell, *nodes)

    def end_cells(self):
        DataHandler.end_cells(self)
        xml_writer.write_footer_cells(self.__ofile)

    def start_meshfunction(self, name, dim, size):
        DataHandler.start_meshfunction(self, name, dim, size)
        fname = os.path.splitext(self.__ofile.name)[0]
        self.__ofile_meshfunc = open("%s_%s.xml" % (fname, name), "w")
        xml_writer.write_header_meshfunction(self.__ofile_meshfunc, dim, size)

    def add_entity_meshfunction(self, index, value):
        DataHandler.add_entity_meshfunction(self, index, value)
        xml_writer.write_entity_meshfunction(self.__ofile_meshfunc, index, value)

    def end_meshfunction(self):
        DataHandler.end_meshfunction(self)
        xml_writer.write_footer_meshfunction(self.__ofile_meshfunc)
        self.__ofile_meshfunc.close()
        self.__ofile_meshfunc = None

    def start_domains(self):
        #DataHandler.start_domains(self)
        xml_writer.write_header_domains(self.__ofile)

    def end_domains(self):
        #DataHandler.end_domains(self)
        xml_writer.write_footer_domains(self.__ofile)

    def start_mesh_value_collection(self, name, dim, size, etype):
        DataHandler.start_mesh_value_collection(self, name, dim, size, etype)
        xml_writer.write_header_meshvaluecollection(self.__ofile, name, dim, size, etype)

    def add_entity_mesh_value_collection(self, dim, index, value, local_entity=0):
        DataHandler.add_entity_mesh_value_collection(self, dim, index, value)
        xml_writer.write_entity_meshvaluecollection(self.__ofile, dim, index, value, local_entity=local_entity)

    def end_mesh_value_collection(self):
        DataHandler.end_mesh_value_collection(self)
        xml_writer.write_footer_meshvaluecollection(self.__ofile)

    def close(self):
        DataHandler.close(self)
        if self.__ofile.closed:
            return
        xml_writer.write_footer_mesh(self.__ofile)
        self.__ofile.close()
        if self.__ofile_meshfunc is not None:
            self.__ofile_meshfunc.close()

def gmsh2xml(ifilename, handler):
    print("Converting from Gmsh format (.msh, .gmsh) to DOLFIN XML format")

    # The dimension of the gmsh element types supported here as well as the dolfin cell types for each dimension
    gmsh_dim = {15: 0, 1: 1, 2: 2, 4: 3}
    cell_type_for_dim = {1: "interval", 2: "triangle", 3: "tetrahedron" }
    # the gmsh element types supported for conversion
    supported_gmsh_element_types = [1, 2, 4, 15]

    # Open files
    ifile = open(ifilename, "r")

    # Scan file for cell type
    cell_type = None
    highest_dim = 0
    line = ifile.readline()
    while line:

        # Remove newline
        line = line.rstrip("\n\r")

        # Read dimension
        if line.find("$Elements") == 0:

            line = ifile.readline()
            num_elements = int(line)
            if num_elements == 0:
                _error("No elements found in gmsh file.")
            line = ifile.readline()

            # Now iterate through elements to find largest dimension.  Gmsh
            # format might include elements of lower dimensions in the element list.
            # We also need to count number of elements of correct dimensions.
            # Also determine which vertices are not used.
            dim_count = {0: 0, 1: 0, 2: 0, 3: 0}
            vertices_used_for_dim = {0: [], 1: [], 2: [], 3: []}
            # Array used to store gmsh tags for 1D (type 1/line), 2D (type 2/triangular) elements and 3D (type 4/tet) elements
            tags_for_dim = {0: [], 1: [], 2: [], 3: []}

            while line.find("$EndElements") == -1:
                element = line.split()
                elem_type = int(element[1])
                num_tags = int(element[2])
                if elem_type in supported_gmsh_element_types:
                    dim = gmsh_dim[elem_type]
                    if highest_dim < dim:
                        highest_dim = dim
                    node_num_list = [int(node) for node in element[3 + num_tags:]]
                    vertices_used_for_dim[dim].extend(node_num_list)
                    if num_tags > 0:
                        tags_for_dim[dim].append(tuple(int(tag) for tag in element[3:3+num_tags]))
                    dim_count[dim] += 1
                else:
                    #TODO: output a warning here. "gmsh element type %d not supported" % elem_type
                    pass
                line = ifile.readline()
        else:
            # Read next line
            line = ifile.readline()

    # Check that we got the cell type and set num_cells_counted
    if highest_dim == 0:
        _error("Unable to find cells of supported type.")

    num_cells_counted = dim_count[highest_dim]
    vertex_set = set(vertices_used_for_dim[highest_dim])
    vertices_used_for_dim[highest_dim] = None

    vertex_dict = {}
    for n,v in enumerate(vertex_set):
        vertex_dict[v] = n

    # Step to beginning of file
    ifile.seek(0)

    # Set mesh type
    handler.set_mesh_type(cell_type_for_dim[highest_dim], highest_dim)

    # Initialise node list (gmsh does not export all vertexes in order)
    nodelist = {}

    # Current state
    state = 0

    # Write data
    num_vertices_read = 0
    num_cells_read = 0

    # Only import the dolfin objects if facet markings exist
    process_facets = False
    if len(tags_for_dim[highest_dim-1]) > 0:
        # first construct the mesh
        try:
            from dolfin import MeshEditor, Mesh
        except ImportError:
            _error("DOLFIN must be installed to handle Gmsh boundary regions")
        mesh = Mesh()
        mesh_editor = MeshEditor ()

        if (highest_dim == 2):
            mesh_editor.open(mesh, "triangle", highest_dim, highest_dim)
        elif (highest_dim == 3):
            mesh_editor.open(mesh, "tetrahedron", highest_dim, highest_dim)
        else:
            raise RuntimeError
        process_facets = True
    else:
        # TODO: Output a warning or an error here
        me = None

    while state != 10:

        # Read next line
        line = ifile.readline()
        if not line: break

        # Skip comments
        if line[0] == '#':
            continue

        # Remove newline
        line = line.rstrip("\n\r")

        if state == 0:
            if line == "$MeshFormat":
                state = 1
        elif state == 1:
            (version, file_type, data_size) = line.split()
            state = 2
        elif state == 2:
            if line == "$EndMeshFormat":
                state = 3
        elif state == 3:
            if line == "$Nodes":
                state = 4
        elif state == 4:
            num_vertices = len(vertex_dict)
            handler.start_vertices(num_vertices)
            if process_facets:
                mesh_editor.init_vertices_global(num_vertices, num_vertices)
            state = 5
        elif state == 5:
            (node_no, x, y, z) = line.split()
            node_no = int(node_no)
            x,y,z = [float(xx) for xx in (x,y,z)]
            if node_no in vertex_dict:
                node_no = vertex_dict[node_no]
            else:
                continue
            nodelist[int(node_no)] = num_vertices_read
            handler.add_vertex(num_vertices_read, [x, y, z])
            if process_facets:
                if highest_dim == 1:
                    coords = numpy.array([x])
                elif highest_dim == 2:
                    coords = numpy.array([x, y])
                elif highest_dim == 3:
                    coords = numpy.array([x, y, z])
                mesh_editor.add_vertex(num_vertices_read, coords)

            num_vertices_read +=1

            if num_vertices == num_vertices_read:
                handler.end_vertices()
                state = 6
        elif state == 6:
            if line == "$EndNodes":
                state = 7
        elif state == 7:
            if line == "$Elements":
                state = 8
        elif state == 8:
            handler.start_cells(num_cells_counted)
            if process_facets:
                mesh_editor.init_cells_global(num_cells_counted, num_cells_counted)

            state = 9
        elif state == 9:
            element = line.split()
            elem_type = int(element[1])
            num_tags  = int(element[2])
            if elem_type in supported_gmsh_element_types:
                dim = gmsh_dim[elem_type]
            else:
                dim = 0
            if dim == highest_dim:
                node_num_list = [vertex_dict[int(node)] for node in element[3 + num_tags:]]
                for node in node_num_list:
                    if not node in nodelist:
                        _error("Vertex %d of %s %d not previously defined." %
                              (node, cell_type_for_dim[dim], num_cells_read))
                cell_nodes = [nodelist[n] for n in node_num_list]
                handler.add_cell(num_cells_read, cell_nodes)

                if process_facets:
                    cell_nodes = numpy.array([nodelist[n] for n in node_num_list], dtype=numpy.uintp)
                    mesh_editor.add_cell(num_cells_read, cell_nodes)

                num_cells_read +=1

            if num_cells_counted == num_cells_read:
                handler.end_cells()
                if process_facets:
                    mesh_editor.close()
                state = 10
        elif state == 10:
            break

    # Write mesh function based on the Physical Regions defined by
    # gmsh, but only if they are not all zero. All zero physical
    # regions indicate that no physical regions were defined.
    if highest_dim not in [1,2,3]:
        _error("Gmsh tags not supported for dimension %i. Probably a bug" % dim)

    tags = tags_for_dim[highest_dim]
    physical_regions = tuple(tag[0] for tag in tags)
    if not all(tag == 0 for tag in physical_regions):
        handler.start_meshfunction("physical_region", dim, num_cells_counted)
        for i, physical_region in enumerate(physical_regions):
            handler.add_entity_meshfunction(i, physical_region)
        handler.end_meshfunction()

    # Now process the facet markers
    tags = tags_for_dim[highest_dim-1]
    if (len(tags) > 0) and (mesh is not None):
        physical_regions = tuple(tag[0] for tag in tags)
        if not all(tag == 0 for tag in physical_regions):
            mesh.init(highest_dim-1,0)

            # Get the facet-node connectivity information (reshape as a row of node indices per facet)
            if highest_dim==1:
              # for 1d meshes the mesh topology returns the vertex to vertex map, which isn't what we want
              # as facets are vertices
              facets_as_nodes = numpy.array([[i] for i in range(mesh.num_facets())])
            else:
                facets_as_nodes =  numpy.reshape(mesh.topology()(highest_dim-1,0)(),(mesh.num_facets(), highest_dim))

            # Build the reverse map
            nodes_as_facets = {}
            for facet in range(mesh.num_facets()):
              nodes_as_facets[tuple(facets_as_nodes[facet,:])] = facet

            data = [int(0*k) for k in range(mesh.num_facets()) ]
            for i, physical_region in enumerate(physical_regions):
                nodes = [n-1 for n in vertices_used_for_dim[highest_dim-1][highest_dim*i:(highest_dim*i+highest_dim)]]
                nodes.sort()

                if physical_region != 0:
                    try:
                        index = nodes_as_facets[tuple(nodes)]
                        data[index] = physical_region
                    except IndexError:
                        raise Exception ( "The facet (%d) was not found to mark: %s" % (i, nodes) )

            # Create and initialise the mesh function
            handler.start_meshfunction("facet_region", highest_dim-1, mesh.num_facets() )
            for index, physical_region in enumerate ( data ):
                handler.add_entity_meshfunction(index, physical_region)
            handler.end_meshfunction()

    # Check that we got all data
    if state == 10:
        print("Conversion done")
    else:
       _error("Missing data, unable to convert \n\ Did you use version 2.0 of the gmsh file format?")

    # Close files
    ifile.close()

keep = input("keep temporary files after conversion? (y/N): ")
if keep == 'y':
    keepmsh = True

# Konvertierungsteil
print('starting conversion...')
msh_path = filename[:-4] + '.msh'

# med2msh mit gmsh
os.system(gmsh_path + ' ' + filename + ' -3 -v 0 -format msh2 ' + msh_path)

objectConvert = XmlHandler(msh_path[:-4] + str(".xml"))
# msh2xml
gmsh2xml(msh_path, objectConvert)
objectConvert.close()

'''
# Notwendiger Footer wurde in xml nicht geschrieben. Als Fix folgende fuenf Zeilen:
with open(msh_path[:-4] + str(".xml"), "a") as myfile:
    myfile.write("""\
  </mesh>
</dolfin>
""")
'''
# Ab hier xml2h5 Konvertierung
mesh = Mesh(msh_path[:-4] + str(".xml"))
hdf = HDF5File(mesh.mpi_comm(),msh_path[:-4]+'.h5','w')
hdf.write(mesh, '/mesh')

try: 
	cells = MeshFunction('size_t', mesh, msh_path[:-4] + '_physical_region.xml' )
	hdf.write(cells, '/cells')
except:
	print('No cells are written! Check the med file if not in purpose!')
	pass

try:
	facets = MeshFunction('size_t', mesh, msh_path[:-4] + '_facet_region.xml' )
	hdf.write(facets, '/facets')
except:
	print('No facets are written! Check the med file if not in purpose!')
	pass

# Erhalten der Mesh Informationen aus Header der .msh Datei
fobj = open(msh_path)
lines=fobj.readlines()

#Betrachte ersten 1000 Zeilen der msh Datei und begrenze zwischen den beiden Strings PhysikalNames u. EndPhy...
#und speichere die begrenzenden Zeilennummern in die Variablen beg und end
try:
    for i in range(1000):
        if lines[i] == '$PhysicalNames\n':
            beg = i
        if lines[i] == '$EndPhysicalNames\n':
            end = i
except IndexError:
    print('no regions found! exiting....')
    exit()

fobj.close()

# erstelle String des relevanten Teils der Subdomain Information aus .msh Header
list_str = ""            
for i in range(beg+2,end):
    list_str += lines[i]
            
print('------------------------------\nDim / ID / Subdomain name\n------------------------------\n' + list_str[:-1])

# Standardeinstellung: Loesche .msh Datei nach Konvertierung. Wenn nicht gewuenscht, Suffix -k nutzen
# Bsp: python3 convert.py <Dateiname> -k
if keepmsh == False:
    os.remove(msh_path)
    os.remove(msh_path[:-4] + str('.xml'))
    try: os.remove(msh_path[:-4] + str('_physical_region.xml'))
    except: pass
    try: os.remove(msh_path[:-4] + str('_facet_region.xml'))
    except: pass
    
txt = open(msh_path[:-4] + str('.txt'), 'w')
txt.write('------------------------------\nDim / ID / Subdomain name\n------------------------------\n' + list_str[:-1])
txt.close()
print('------------------------------\nsubdomain information written in ' + msh_path[:-4] + '.txt')
