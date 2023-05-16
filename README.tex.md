# amorphy

During my PhD tenure, me in collaboration with my supervisor Dr. Assil Bouzid, we have written a code named _amorphy_. The purpose of this code base is to perform structural analysis for amorphous system. The coordination number, local environment, charge analysis etc. reported in this work have been computed using _amorphy_ code.


\\begin{figure}\[!htbp\] \\centering
\\includegraphics\[scale=0.45\]{chapters/images/appendix/amorphy.png}
\\caption{Codes utilized for structural analysis of amorphous system}
\\end{figure} \\ 

The github link to [_amorphy_] (https://github.com/rvraghvender/amorph "amorphy GitHub")

## {Description of _amorphy_ code

**inputValues.py:** Implements the function for computing desired properties and takes the input parameters such as cut-offs, atomic symbols etc.  

**read_trajectory.py:** This module read/load the MD trajectory (xyz format). 

**bond_asymmetry.py:** This module computes the asymmetry in bond-length around given atom. 
**geometric_properties.py:** This module performs the geometrical analysis such as identifying the shape of simulation cell (orthorhombic/ non-orthorhombic), rotate the
molecule etc. 

**periodic_boundary_conditions.py:** This module helps in implementing the periodic boundary conditions in orthorhombic or non-orthorhombic systems. 

**charge_analysis.py:** This module provides an interface to the charges obtained from DDEC6 or Bader analysis.

**radial_and_bond_angle_distribution.py:** This module calculates the following: 
- Radial pair distance distribution: In this submodule, distance distribution between two atomic species is calculated.
- Radial distribution function (RDF): In this submodule, partial PDF is calculated between two atomic species.
- Bond-angle distribution function (BAD): In this submodule, bond-angle distribution of angle between atoms to and is calculated.

**topology.py:** This module performs the following operations:
- Compute coordination: In this submodule, the coordination number for given atom under cut-off value is calculated. 
- Wrap atoms: This submodule, wraps the atoms placed outside the simulation cell into the box using periodic boundary conditions. 
- Compute all distances: This submodule, prints all the possible distance between two atomic species for each frame. 
- Neighbor list: This submodule, returns a list of atomic ID around given atom. 
- Get constrained frame ID: This submodule, returns the frame ID in which two atoms come closer than the cut-off values defined.
- Hydrogen passivate: This submodule, attaches hydrogen atom to the terminal oxygen atoms in a molecular fragment. 
- Ground the molecule: Given atomic ID of three atoms, this submodule, translates and rotates the whole molecular fragment in such a way that the given IDs are placed on the z = 0 plane.
- Compute BO/NBO coordination: This submodule, computes the coordination number of cation with respect to bridging oxygen and non-bridging oxygen separately.

**wannier_structural_analysis.py:** This module performs the following operations:
- Use wannier centers around cation to study its local environment: Here considering cation as a host atom, the local environments such coordination number resolved
percentage, average charge, bond asymmetry etc. around cation are obtained under the applied Wannier center constraints. 
- Use wannier centers around anion to study its local environment: Here considering anion as a host atom, the local environments such as coordination number resolved percentage, average charge, bond asymmetry etc. around anion are obtained under the applied Wannier center constraints. 
- Get BO/NBO ID: In this submodule, the atomic ID of bridging and non-bridging oxygen are returned. 

