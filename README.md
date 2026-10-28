<h1>Computational Reality</h1>

<p> This repository is helping to do computations explained in the book <a href="https://link.springer.com/book/10.1007/978-981-10-2444-3" target="_blank" class="external" rel="noreferrer noopener"><span>Computational Reality</span></a>

<p>How to install SciPy, Docker, FEniCS, ParaView locally and use them?
  
  
<h2>Choose your editor</h2>
<p>We need to have an editor for writing any kind of code for 
computations. A good editor makes the life much easier. There are very 
simple yet powerful editors, using colors and brackets matching to help 
you to write correctly, for example</p>
<p>Gedit (all OSes)&nbsp; <a href="https://wiki.gnome.org/Apps/Gedit" target="_blank" class="external" rel="noreferrer noopener"><span>https://wiki.gnome.org/Apps/Gedit</span></a>

<p>Notepad++ (only Windows)&nbsp; <a href="https://notepad-plus-plus.org/" target="_blank" class="external" rel="noreferrer noopener"><span>https://notepad-plus-plus.org/</span></a>

<p>Of course there are more sophisticated and very elegant editors 
(allowing to do crazy nice editing as well as really helping to write by
 showing variables and so on), for example</p>
<p>Sublime (all OSes) <a href="https://www.sublimetext.com/" target="_blank" class="external" rel="noreferrer noopener"><span>https://www.sublimetext.com/</span></a>

<p>Atom (all OSes) <a href="https://atom.io/" target="_blank" class="external" rel="noreferrer noopener"><span>https://atom.io/</span></a>

<p>Visual Studio (all OSes) <a href="https://code.visualstudio.com/Download" target="_blank" class="external" rel="noreferrer noopener"><span>https://code.visualstudio.com/Download</span></a>

<p>&nbsp;</p>
<h2>Docker</h2>
<p>A modern way of using new apps or softwares is to start them in a 
container. A container is technically a self running system (normally a 
small Linux machine) with all necessary installations and setups have 
been realized. The good part of a container is that the user does not 
install anything to the system, but downloads a huge container and runs 
this in the computer. The prominent container is docker, see more 
details under <a class="instructure_file_link inline_disabled external" href="https://www.docker.com/why-docker" target="_blank" rel="noreferrer noopener"><span>Docker web site.</span></a> We will use FEniCS in a docker container such that first Docker Desktop needs to be installed,</p>
<p>Windows <a class="instructure_file_link inline_disabled external" href="https://desktop.docker.com/win/stable/Docker%20Desktop%20Installer.exe" target="_blank" rel="noreferrer noopener"><span>https://desktop.docker.com/win/stable/Docker%20Desktop%20Installer.exe</span></a>
</span></a>&nbsp;</p>
<p>MacOS <a class="instructure_file_link inline_disabled external" href="https://desktop.docker.com/mac/stable/Docker.dmg" target="_blank" rel="noreferrer noopener"><span>https://desktop.docker.com/mac/stable/Docker.dmg</span></a>
</span></a>&nbsp;</p>
<p>Linux is where also Docker container is created such that there is not a Docker Desktop but a <a class="instructure_file_link inline_disabled external" href="https://hub.docker.com/search?q=&amp;type=edition&amp;offering=community&amp;operating_system=linux" target="_blank" rel="noreferrer noopener"><span>Docker Engine version to install</span></a>
</span></a>. For example, step-by-step instruction for Ubuntu version <a class="instructure_file_link inline_disabled external" href="https://docs.docker.com/engine/install/ubuntu/" target="_blank" rel="noreferrer noopener"><span>https://docs.docker.com/engine/install/ubuntu/</span></a>
</span></a>&nbsp;</p>
<p>After installing, open a PowerShell (windows), shell or terminal (linux, macos) and type</p>
<pre>docker version</pre>
<p>to check the version as well as type</p>
<pre>docker run hello-world</pre>
<p>for verifying if the docker can connect, pull a container, and print into the shell hello world!</p>
<p>&nbsp;</p>
<h2>Running FEniCS in Docker</h2>
<p>For the direct problem, we will use finite element method. There are 
commercial software solutions, yet it is often challenging to implement 
state-of-the-art algorithms. One of the superior development as an 
open-source project is FEniCS and we will use it in this course.</p>
<p>In order to download the latest stable version of FEniCs, type in the shell</p>
<pre>docker pull <code>ghcr.io/scientificcomputing/fenics:2024-02-19</code></pre>
<p>This will take a couple of minutes, since a huge container of many MBs are downloaded. For starting FEniCS, run in the shell</p>
<pre>docker run -ti <code>ghcr.io/scientificcomputing/fenics:2024-02-19</code></pre>
<p>A short welcoming message like that</p>
<p># FEniCS stable version image<br>Welcome to FEniCS/stable!</p>
<p>If necessary, use "ctrl + c" to get back the control. Then there is a prompt where you can run FEniCS</p>
<pre>fenics@0521831b5f28:~$</pre>
<p>Before @ sign, fenics is the username inside the container. It is 
basically a Ubuntu OS (operating system) with FEniCS installed on it. 
After the @ sign the number 0521831b5f28 is the CONTAINER ID that Docker
 has assigned. This ID is unique to the installed computer within the 
container and will be different in each one's own system. All Linux 
commands are useful</p>
<pre>ls</pre>
<p>for seeing a list of the directory</p>
<pre>ls -all</pre>
<p>for more details</p>
<pre>pwd</pre>
<p>to show the present working directory</p>
<pre>cd</pre>
<p>for changing directory</p>
<pre>rm</pre>
<p>to remove a file.</p>
<p>Use the command</p>
<pre>exit</pre>
<p>for exiting the container and going back to the shell of the OS.</p>
<h2>Sharing files from the host into the container</h2>
<p>The container is an isolated system from the OS (operating system) 
such that no harm can be made to the original system. In other words, we
 need to link directories between the OS and the container for a 
possible exchange. Otherwise, the container cannot read or write to the 
files present in the OS.</p>
<p>Open the PowerShell and change the directory to the current (desired)
 directory for linking, for example a directory on the desktop in 
windows (username is "johndoe" as an example)</p>
<pre>cd C:\Users\johndoe\Desktop\compreal</pre>
<p>Windows users may need to manually add this directory in docker.<br>Open Docker: Setting &gt; Resources &gt; File sharing &gt; manually add the directory</p>
<p>In Linux (Ubuntu, Fedora, MacOS) open a terminal and use for the user "johndoe"</p>
<pre>cd /home/johndoe/Desktop/compreal</pre>
<p>By using mkdir this directory can be created first (or using Windows Explorer, MacOS Finder and mouse clicks).</p>
<p>Now, being in the current directory, run the below command for sharing files between the OS and the container</p>
<p>Windows:</p>
<pre>docker run -ti -v ${PWD}:/home/fenics/shared <code>ghcr.io/scientificcomputing/fenics:2024-02-19</code></pre>
<p>Linux (Ubuntu, Fedora, RedHat, OpenSUSE, ...):</p>
<pre>docker run -ti -v ${pwd}:/home/fenics/shared <code>ghcr.io/scientificcomputing/fenics:2024-02-19</code></pre>
<p>Linux (MacOS):</p>
<pre>docker run -ti -v $(pwd):/home/fenics/shared <code>ghcr.io/scientificcomputing/fenics:2024-02-19</code></pre>
<p>Then, the present working directory (pwd) that is the directory 
compreal in the Desktop will be the same as the shared directory. If 
{PWD} is not working, (PWD) may do the job, or the whole directory can 
be written directly. This means that whatever change we do in shared 
within the container is also changed in the directory compreal and vice 
versa. Now, let us change directory to the shared</p>
<pre>cd /home/fenics/shared</pre>
<p>and create a new file helloworld.py in the Optimization directory with the sophisticated content</p>
<pre>print('hello world!')</pre>
<p>and then run it in python</p>
<pre>python3 helloworld.py</pre>
<p>The code is up and running in a FEniCS container in the shell.</p>
<h2>Installing new packages into the container</h2>
<p>This container is an operating system (it is a Ubuntu OS) and any 
software may be installed. We will need to use matplotlib and tex for 
coding and plotting. Installing is done by using sudo (super user do) 
and then apt (this is the repository where all packages are listed). 
Simply run first</p>
<pre>apt update</pre>
<p>for getting the current list of packages from the internet. Then use</p>
<pre>apt install python3-matplotlib texlive texlive-latex-extra python3-sklearn libglu1-mesa cm-super dvipng</pre>
<p>all these will be downloaded and installing in the correct position. 
Now we can run different codes. The benefit of docker is that once you 
close the container, all changes are lost. So it is difficult to brake 
the system. But this also means that once we exit this container, then 
all installed packages are lost. In order to commit these changes, open 
another powershell. Now we have two shells. On one the container is 
running and on the other shell we can get the info of this container by 
running</p>
<pre>docker ps</pre>
<p>which gives us the running container's ID. Use this ID to commit the changes</p>
<pre>docker commit ID <code>ghcr.io/scientificcomputing/fenics:2024-02-19</code></pre>
<p>It is also possible to create a new image (another 2 GB will be used)</p>
<pre>docker commit ID mynewcontainer</pre>
<p>After committing to the existing image or creating a new image, we 
can exit the container and start again next time from the new image</p>
<p>Windows:</p>
<pre>docker run -ti -v ${PWD}:/home/fenics/shared mynewcontainer</pre>
<p>Linux (Ubuntu, Fedora, RedHat, OpenSUSE, ...):</p>
<pre>docker run -ti -v ${pwd}:/home/fenics/shared mynewcontainer</pre>
<p>Linux (MacOS):</p>
<pre>docker run -ti -v $(pwd):/home/fenics/shared mynewcontainer</pre>
<p>Every time a new container will be started, whatever is saved in the 
shared directory will remain. The rest of changes will be erased unless 
we commit them to the image.</p>
<h2>ParaView</h2>
<p>Probably the best visualization tool for results is ParaView that we also heavily use in this course, see <a class="instructure_file_link inline_disabled external" href="https://www.paraview.org/download/" target="_blank" rel="noreferrer noopener"><span>installation details</span>
</span></a> in all operating systems.</p>
<h2>FreeCAD</h2>
<p>A modern computer aided design (CAD) software, mesh can be exported 
in FEniCS compatible xml file, use the instructions to download on the 
web site <a class="inline_disabled external" href="http://freecad.org/" target="_blank" rel="noreferrer noopener"><span>freecad.org</span></a>

<h2>Salome</h2>
<p>Another CAD software is Salome, the mesh is exported as a med file 
and then converted by a python file, see instructions how to install on 
the web site <a href="https://www.salome-platform.org/" target="_blank" class="external" rel="noreferrer noopener"><span>https://www.salome-platform.org/</span></a>

<p>For running the med to h5 converter, gmsh (Linux version) needs to be downloaded from here <a href="https://gmsh.info/" target="_blank" class="external" rel="noreferrer noopener"><span>https://gmsh.info/</span>
</span></a> and extracted next to the converter python file in the shared folder in docker. Then in the converter file the line with&nbsp;</p>
<div style="color: #000000; background-color: #ffffff; font-family: 'Droid Sans Mono', 'monospace', monospace; font-size: 14px; line-height: 19px; white-space: pre;">
<div><span style="color: #001080;">gmsh_path</span><span style="color: #000000;"> = </span><span style="color: #a31515;">'./gmsh-4.7.1-Linux64/bin/gmsh'</span></div>
</div>
<p>needs to be amended with the version downloaded. In this way, running the converter</p>
<pre>python3 convert_med2h5.py meshfromsalome.med</pre>
<p>with the argument of Salome generated med will create a h5 file used in some examples.</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>***********************************************************<br>Useful Links:<br><a class="instructure_file_link inline_disabled external" href="https://fenicsproject.org/download/" target="_blank" rel="noreferrer noopener"><span>https://fenicsproject.org/download/</span>

<p><a class="instructure_file_link inline_disabled external" href="https://readthedocs.org/projects/fenics-containers/downloads/pdf/latest/" target="_blank" rel="noreferrer noopener"><span>https://readthedocs.org/projects/fenics-containers/downloads/pdf/latest/</span>

<p><a class="instructure_file_link inline_disabled external" href="https://readthedocs.org/projects/fenics-containers/downloads/pdf/latest/" target="_blank" rel="noreferrer noopener"><span></span>
</span></a><a href="https://fenicsproject.discourse.group/" target="_blank" class="external" rel="noreferrer noopener"><span>https://fenicsproject.discourse.group/</span>

<p>&nbsp;</p>
<p>Other installation methods:<br><a class="instructure_file_link inline_disabled external" href="https://fenicsproject.org/download/" target="_blank" rel="noreferrer noopener"><span>https://fenicsproject.org/download/</span>

