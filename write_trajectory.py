import numpy as np
from read_trajectory import Trajectory
from periodic_boundary_condition import displacement
from progress_bar import ShowBar, ProgressBar
from inputValues import fileTraj, SKIP, RESOLUTION, A, B, C, Directory
from elemental_data import atomic_symbol, atomic_no
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

        

    def categorize_BO_NBO_atoms(self):
        '''This functions rewrites the trajectory by categorizing O atom 
           into BO and NBO atoms.

        '''
        from wannier_structural_analysis import WannierAnalysis
        TeO2 = WannierAnalysis()
        BO_ID, NBO_ID = TeO2.compute_neighbour_wannier_host_anion()[1:]

        outputfile = Directory + 'split_BO_NBO_traj.xyz'
        with open(outputfile, 'w') as fw:

            for step in tqdm(range(self.n_steps)):
                
                coordinates = np.copy(self.coordinates[step])
                fw.write(f'{len(coordinates)} \n')
                fw.write(f' Lattice :  {A}  {B}  {C} \n')

                for atom_ID, value in enumerate(coordinates):
                    if value[0] != atomic_no('O'):
                        fw.write(f' {str(atomic_symbol(value[0])):>2}    {value[1]:>15.10f}   {value[2]:>15.10f}   {value[3]:>15.10f} \n')
                    
                    elif atom_ID in BO_ID[step][0]:
                        fw.write(f' {str("OB")}    {value[1]:>15.10f}   {value[2]:>15.10f}   {value[3]:>15.10f} \n')
                    
                    elif atom_ID in NBO_ID[step][0]:
                        fw.write(f' {str("ON")}    {value[1]:>15.10f}   {value[2]:>15.10f}   {value[3]:>15.10f} \n')
