import numpy as np
from read_trajectory import Trajectory
from periodic_boundary_condition import displacement
from progress_bar import ShowBar, ProgressBar
from inputValues import fileTraj, SKIP, RESOLUTION, A, B, C
from elemental_data import atomic_symbol
from tqdm import tqdm

class WriteTrajectory(Trajectory):
    def __init__(self, filename=fileTraj, skip=SKIP, resolution=RESOLUTION):
        Trajectory.__init__(self, filename=filename, skip=skip, resolution=resolution)
 

    def sequencing_trajectory(self):
        '''Trajectory file written with decreasing atomic number.
        '''
        
        outputfile = '.'.join(fileTraj.split('.')[:-1])+'_sequence_traj.xyz'
        with open(outputfile, 'w') as fw:
            for step in tqdm(range(self.n_steps)):
                coordinates = np.copy(self.coordinates[step])
                coordinates = coordinates[coordinates[:,0].argsort()]
                fw.write(f' {len(coordinates)} \n  {A}  {B}  {C} \n')
                for value in coordinates[::-1]:
                    fw.write(f' {atomic_symbol(value[0]):>2}    {value[1]:>8.6f}    {value[2]:>8.6f}    {value[3]:>8.6f} \n')

        

    def clean_degenerate_wannier_atoms(self, rcut=0.8):
        '''This functions takes a particular atom and search for the farthest wannier
           atom with the symbol X located within cutoff r. And thus write the traject-
           ory with just non-generate atoms.

           Limitation : Currently I assume that it is possible to find a common cutoff
           of all the elements without X in the trajectory file.
        '''
        outputfile = '.'.join(fileTraj.split('.')[:-1])+'_cleaned_traj.xyz'
        with open(outputfile, 'w') as fw:

            fw.write(f' REPLACE \n \n')
            for step in range(self.n_steps):
                progressBar = "\rProgress: " + ProgressBar(self.n_steps -1 , step, 100)
                ShowBar(progressBar)


                count=0

                Atom_ID = 0
                for i, atom in enumerate(self.coordinates[step]):
                    
                    Atom_ID += 1
                    if atom[0] == 'Tl':
                        coord_atom = np.array(atom[1:], dtype=float)
                        ener = np.float64(self.coordinates_energy[step][Atom_ID -1])
                        fw.write(f' {atom[0]:2s}      {atom[1]:12f}      {atom[2]:12f}      {atom[3]:12f}      {ener:12f}\n')
                        count += 1

                        coord_of_various_wannier_centers_near_host_atom = []
                        dist_of_various_wannier_centers_near_host_atom = []
                        ener_of_various_wannier_centers_near_host_atom = []
                        
                        Atom_ID_wannier = 0
                        for j, atom_wannier in enumerate(self.coordinates[step]):
                           
                            Atom_ID_wannier += 1
                            if atom_wannier[0] == 'X':
                                coord_atom_wannier = np.array(atom_wannier[1:], dtype=float)
                                ener = np.float(self.coordinates_energy[step][Atom_ID_wannier -1])
                                _, dist_atom_wannier_near_host = displacement(coord_atom, coord_atom_wannier)
    
                                # I assume no oxygen-wanniers are with in radium 1 Ang of host atom
                                if dist_atom_wannier_near_host < rcut:   # Cutoff for Te-W
                                    coord_of_various_wannier_centers_near_host_atom.append(coord_atom_wannier)
                                    dist_of_various_wannier_centers_near_host_atom.append(dist_atom_wannier_near_host)
                                    ener_of_various_wannier_centers_near_host_atom.append(ener)
    
                        coord_atom_wannier_near_host = coord_of_various_wannier_centers_near_host_atom[dist_of_various_wannier_centers_near_host_atom.index(max(dist_of_various_wannier_centers_near_host_atom))]
                        ener = np.float64(ener_of_various_wannier_centers_near_host_atom[dist_of_various_wannier_centers_near_host_atom.index(max(dist_of_various_wannier_centers_near_host_atom))])
                        fw.write(f'  X      {coord_atom_wannier_near_host[0]:12f}      {coord_atom_wannier_near_host[1]:12f}      {coord_atom_wannier_near_host[2]:12f}      {ener:12f}\n ')
                        count += 1

                    elif atom[0] != 'Tl' and atom[0] != 'X':
                        ener = np.float64(self.coordinates_energy[step][Atom_ID -1])
                        coord_atom = np.array(atom[1:], dtype=float)
                        fw.write(f' {atom[0]:2s}      {atom[1]:12f}      {atom[2]:12f}      {atom[3]:12f}     {ener:12f} \n')
                        count += 1
                        
                        Atom_ID_wannier = 0
                        for j, atom_wannier in enumerate(self.coordinates[step]):

                            Atom_ID_wannier += 1
                            if atom_wannier[0] == 'X':
                                coord_atom_wannier = np.array(atom_wannier[1:], dtype=float)
                                _, dist_atom_wannier_near_host = displacement(coord_atom, coord_atom_wannier)
                                ener = np.float64(self.coordinates_energy[step][Atom_ID_wannier-1]) 
                              
                                # I assume no oxygen-wanniers are with in radium 1 Ang of host atom
                                if dist_atom_wannier_near_host < rcut:   # Cutoff for Te-W
                                    fw.write(f' {atom_wannier[0]:2s}      {atom_wannier[1]:12f}      {atom_wannier[2]:12f}      {atom_wannier[3]:12f}     {ener:12f} \n')
                                    count += 1
                fw.write(f' {count} \n \n')
