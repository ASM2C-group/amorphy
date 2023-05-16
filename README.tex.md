# amorphy

\\textit{amorphy}}\\label{ann_raman_spectroscopy} During my PhD tenure,
me in collaboration with my supervisor Dr. Assil Bouzid, we have written
a code named \\textit{amorphy}\'\'. The purpose of this code base is to
perform structural analysis for amorphous system. The coordination
number, local environment, charge analysis etc. reported in this work
have been computed using \\textit{amorphy}\'\' code.
\\begin{figure}\[!htbp\] \\centering
\\includegraphics\[scale=0.45\]{chapters/images/appendix/amorphy.png}
\\caption{Codes utilized for structural analysis of amorphous system}
\\end{figure} \\ The github link to \`\`\\textit{amorphy}\'\' :
\\url{https://github.com/rvraghvender/amorphy}

\\subsection\*{Description of \\textit{amorphy} code} \\begin{itemize}
\\item \\textbf{inputValues.py:} Implements the function for computing
desired properties and takes the input parameters such as cut-offs,
atomic symbols etc. \\item \\textbf{read_trajectory.py:} This module
read/load the MD trajectory (xyz format). \\item
\\textbf{bond_asymmetry.py:} This module computes the asymmetry in
bond-length around given atom. \\item \\textbf{geometric_properties.py:}
This module performs the geometrical analysis such as identifying the
shape of simulation cell (orthorhombic/ non-orthorhombic), rotate the
molecule etc. \\item \\textbf{periodic_boundary_conditions.py:} This
module helps in implementing the periodic boundary conditions in
orthorhombic or non-orthorhombic systems. \\item
\\textbf{charge_analysis.py:} This module provides an interface to the
charges obtained from DDEC6 or Bader analysis. \\item
\\textbf{radial_and_bond_angle_distribution.py:} This module calculates
the following: \\begin{itemize} \\item Radial pair distance
distribution: In this submodule, distance distribution between two
atomic species is calculated. \\item Radial distribution function: In
this submodule, partial PDF is calculated between two atomic species.
\\item Bond-angle distribution function (BAD): In this submodule,
bond-angle distribution of angle between atoms to and is calculated.
\\end{itemize} \\item \\textbf{topology.py:} This module performs the
following operations: \\begin{itemize} \\item Compute coordination: In
this submodule, the coordination number for given atom under cut-off
value is calculated. \\item Wrap atoms: This submodule, wraps the atoms
placed outside the simulation cell into the box using periodic boundary
conditions. \\item Compute all distances: This submodule, prints all the
possible distance between two atomic species for each frame. \\item
Neighbor list: This submodule, returns a list of atomic ID around given
atom. \\item Get constrained frame ID: This submodule, returns the frame
ID in which two atoms come closer than the cut-off values defined.
\\item Hydrogen passivate: This submodule, attaches hydrogen atom to the
terminal oxygen atoms in a molecular fragment. \\item Ground the
molecule: Given atomic ID of three atoms, this submodule, translates and
rotates the whole molecular fragment in such a way that the given IDs
are placed on the z = 0 plane. \\item Compute BO/NBO coordination: This
submodule, computes the coordination number of cation with respect to
bridging oxygen and non-bridging oxygen separately. \\end{itemize}
\\item \\textbf{wannier_structural_analysis.py:} This module performs
the following operations:
- Use wannier centers around cation to study its local environment: Here considering cation as a host atom, the local environments such coordination number resolved
percentage, average charge, bond asymmetry etc. around cation are obtained under the applied Wannier center constraints. 
- Use wannier centers around anion to study its local environment: Here considering anion as a host atom, the local environments such as coordination number resolved percentage, average charge, bond asymmetry etc. around anion are obtained under the applied Wannier center constraints. 
- Get BO/NBO ID: In this submodule, the atomic ID of bridging and non-bridging oxygen are returned. 
\\end{itemize}
