from read_trajectory import Trajectory
from tqdm import tqdm
from matplotlib import pyplot as plt
from  smooth_2D_data import smooth_data
import numpy as np
import matplotlib_style
from progress_bar import ShowBar, ProgressBar
from geometric_properties import volume_sphere
from elemental_data import atomic_no, atomic_symbol
from periodic_boundary_condition import displacement, angle
from inputValues import (fileTraj, SKIP, RESOLUTION, atom_name_2, atom_name_1, atom_name_3, 
                        A, B, C, bond_len_cut_pair12, bond_len_cut_pair23, Directory)



class PDF(Trajectory):
    def __init__(self, filename=fileTraj, skip=SKIP, resolution = RESOLUTION):
        Trajectory.__init__(self, filename=filename, skip=skip, resolution = resolution)

    def compute_volume_per_atom(self): 

            # Collecting number of desired atoms in whole non-skipped trajectory
            n_desired_atom = 0 
            for step in range(self.n_steps): 
                for i, atom in enumerate(self.coordinates[step]):
                    if self.atom_list[i] == atom_name_2: 
                        n_desired_atom += 1 
                        
            # Volume of Simulation cell            
            volume_of_cell = A * B * C  
            
            # This will be usefule for subtracting out the number of host atoms while normalisation.  
            # For instance, if I am looking for Te-Te RDF then it will subtract one atom in each 
            # configuration in nstep. Thus nothing will be removed when both atom differs.
            if atom_name_1 == atom_name_2:
                remove = 1   
            else:  
                remove = 0 

            # This is number of atoms to be divide while normalizing. Example for same atoms (Te:Te),
            # selection it will remove n_steps number of atoms from total selected atom in trajectory 
            # and will later divied by n_steps to find average number of atoms in each step. For different
            # selection of atoms, remove = 0, thus, average_n_desired_atom = number of selected atom in 
            # each configuration.
            average_n_desired_atom = (n_desired_atom - self.n_steps * remove) / self.n_steps   
            
            self.compute_volume_per_atom = volume_of_cell / average_n_desired_atom

    def compute_radial_distribution(self, r_cutoff=0.5*min(A,B,C)):
        
        # Thickness of bin 
        dr = r_cutoff / self.resolution 
                
        self.radii = np.linspace(0.010, r_cutoff, self.resolution)  
        volumes = np.zeros(self.resolution) 
        self.g_of_r = np.zeros(self.resolution) 

        for step in tqdm(range(self.n_steps)):
            
            # Printng live stepcount and total stepcount
            #     print(" Total Steps : {} ".format(self.n_steps), end='\r') 
            #progressBar = "\rProgress: " + ProgressBar(self.n_steps-1, step, 100)
            #ShowBar(progressBar)
            
            # coord_atoms_1 represent empty list for host and non-host atoms respectively.
            coord_atoms_1 = [] 
            coord_atoms_2 = [] 
            
            for i, atom in enumerate(self.coordinates[step]):  
                if self.atom_list[i] == atom_name_1 : 
                    coord_atoms_1.append(atom[1:])  
                if self.atom_list[i] == atom_name_2 : 
                    coord_atoms_2.append(atom[1:])

            coord_atoms_1 = np.array(coord_atoms_1, dtype=np.float_) 
            coord_atoms_2 = np.array(coord_atoms_2, dtype=np.float_) 

            for i, atom1 in enumerate(coord_atoms_1):
                for j in range(self.resolution):
                    r1 = ( j + 0.010 )  * dr   # lower radius of the shell
                    r2 = r1 + dr               # higher radius of the shell
                    v1 = volume_sphere(r1, A, B, C) 
                    v2 = volume_sphere(r2, A, B, C)
                    
                    # Volume of spherical shell stored in a list
                    volumes[j] += v2 - v1 

                for k, atom2 in enumerate(coord_atoms_2):
                    _, dist = displacement(atom1, atom2)
                    
                    # Gives the integer value of distance
                    index = int(dist / dr) 

                    # Here only atoms with rcutoff are considered. Furthermore, each time atoms located 
                    # index distance are found value of self.g_of_r is added.
                    if 0 < index < self.resolution: 
                        self.g_of_r[index] += 1.0 

        for i, value in enumerate(self.g_of_r):
            
            # Here number of atoms at certain index is multiplied by volume per unit atom
            # and later divided by volume of shell at given index.
            self.g_of_r[i] = value * self.compute_volume_per_atom / volumes[i]

    def radial_distribution_histogram(self, atom_name_1, 
                                            atom_name_2, 
                                            binwidth , 
                                            normalized=False,
                                            smooth_data=False):
        '''
           This function evaluates the histogram of radial distances between pair of atoms.
           Generally, this function becomes useful in the case of calculating atom-wannier 
           distance distributions in order to evaluate the cutoff for defining coordinaton 
           number.

           Parameters:
           ------------------------------------------------------------------------------
           atom_name_1 : Atom name symbol of atom 1
           atom_name_2 : Atom name symbol of atom 2
           binwidth    : Width of bin for histogram
           normalized  : Normalize the data with n_steps and count of atom_name_1
           smooth_data : Apply gaussian smootheing to 2D data
           
           Return:
           ------------------------------------------------------------------------------
        '''

        if isinstance(atom_name_1, str):
            atom_name_1 = atomic_no(atom_name_1)
        if isinstance(atom_name_2, str):
            atom_name_2 = atomic_no(atom_name_2)

        Distances = []
        count_atom_name_1 = self.atom_list.count(atomic_symbol(atom_name_1))
        
        for step in tqdm(range(self.n_steps)):
            
            for atomID_1, value_1 in enumerate(self.coordinates[step]):
                 if atom_name_1 == value_1[0]:

                     for atomID_2, value_2 in enumerate(self.coordinates[step]):
                         if atom_name_2 == value_2[0]:
                             
                             _, dist = displacement(value_1[1:], value_2[1:])
                             Distances.append(float(dist))

        xmin = min(Distances)
        xmax = max(Distances)
        bins = int((xmax-xmin)/binwidth)
        
        with open(Directory+f'{atomic_symbol(atom_name_1)}-{atomic_symbol(atom_name_2)}-disthist.dat', 'w') as fw:

            y, edges = np.histogram(Distances, bins=bins)
            y = y/ self.n_steps
            centers = 0.5*(edges[1:] + edges[:-1])
            if smooth_data:
                centers, y = smooth_data(y, centers)
            
            if normalized:
                y = y / (self.n_steps * count_atom_name_1) 

            fw.write('# Distance   Hist')
            for i in range(len(y)):
                fw.write(f'\n  {centers[i]:>8.5f}    {y[i]:>8.5f} ')


    #################################################################################################################

    def compute_bond_angle_distribution(self):
        
        self.angle = np.linspace(0.0, 180, 180, dtype=int)
        self.bdf = np.zeros(180, dtype=np.float64)
        
        for step in tqdm(range(self.n_steps)):
            angle_data_atom_1 = []
            angle_data_atom_2 = []
            angle_data_atom_3 = []
            
            # Printing live stepcount and total stepcound
            #             print(f" Total Steps : {self.n_steps} ", end='\r')  
            #progressBar = "\rProgress: " + ProgressBar(self.n_steps -1 , step, 100)
            #ShowBar(progressBar)
            
            for i, atom in enumerate(self.coordinates[step]):
                if self.atom_list[i] == atom_name_1 :
                    angle_data_atom_1.append(atom[1:])
                if self.atom_list[i]==  atom_name_2 :
                    angle_data_atom_2.append(atom[1:])
                if self.atom_list[i]==  atom_name_3 :
                    angle_data_atom_3.append(atom[1:])
            angle_data_atom_1 = np.array(angle_data_atom_1, dtype=np.float64)
            angle_data_atom_2 = np.array(angle_data_atom_2, dtype=np.float64)
            angle_data_atom_3 = np.array(angle_data_atom_3, dtype=np.float64)

            for i, atom1 in enumerate(angle_data_atom_1):
                for j, atom2 in enumerate(angle_data_atom_2):
                    r12, rij = displacement(atom2,atom1)
                    
                    
                    if 0.0 < rij <= bond_len_cut_pair12:
                        for k, atom3 in enumerate(angle_data_atom_3):
                            r32, rjk = displacement(atom2,atom3)
                            
                            if 0.0 < rjk <= bond_len_cut_pair23:
                                degree_angle = angle(atom1, atom2, atom3)
                                angle_index = int(degree_angle)
                                if angle_index != 0:
                                    self.bdf[angle_index] += 1.0

        for i, value in enumerate(self.bdf):
            self.bdf[i] = value  / ( self.n_steps )

    def bond_plot(self, filename="", savefig=False, write_data=False):

        if not self.g_of_r.any():
            print('compute the radial distribution function first\n')
            return

        if write_data:
            with open(Directory+'bond_plot.dat', 'w') as fw:
                fw.write(f'# Radii \t gr[{atom_name_1}-{atom_name_2}]  \n')
                for i in range(len(self.radii)):
                    fw.write(f'{self.radii[i]} \t {self.g_of_r[i]} \n')

        plt.figure(figsize=[16,12])
        plt.title('Radial Distribution Function of TeO$_{2}$ glass', fontdict={'fontsize':18})
        plt.xlabel('r (A$^0$)', fontdict={'fontsize':15})
        plt.ylabel('g$_{ab}$(r)', fontdict={'fontsize':15})
        plt.xticks(np.arange(0, max(self.radii)+1, step=0.5), fontsize=15)
        plt.yticks(fontsize=15)
        plt.plot(self.radii, self.g_of_r,label=(f'{atom_name_1}-{atom_name_2}'))
        plt.legend(prop={"size":15})
        plt.show()
        if savefig: plt.savefig(Directory+'g_of_r_{}-{}.png'.format(atom_name_1,atom_name_2), dpi=300, bbox='tight', format='png')
        plt.clf()


    def bond_angle_plot(self, filename ="", savefig=False, write_data=False):

        if not self.bdf.any():
            print('compute the bond-angle distribution function first\n')
            return

        if write_data:
            with open(Directory+'bond_angle_plot.dat', 'w') as fw:
                fw.write(f'# Angle \t bdf[{atom_name_1}-{atom_name_2}-{atom_name_3}]  \n')
                for i in range(len(self.angle)):
                    fw.write(f'{self.angle[i]} \t {self.bdf[i]} \n')

        plt.figure(figsize=[16,12])
        plt.title('Bond-angle Distribution Function of TeO$_{2}$ glass', fontdict={'fontsize':18})
        plt.xlabel('Theta (degrees) ', fontdict={'fontsize':15})
        plt.ylabel(' f(theta) ', fontdict={'fontsize':15})
        plt.plot(self.angle, self.bdf,label = f'{atom_name_1}-{atom_name_2}-{atom_name_3}')
        plt.legend(prop={"size":15})
        plt.show()
        if savefig: plt.savefig(Directory+'bdf_{}-{}-{}.png'.format(atom_name_1,atom_name_2,atom_name_3), dpi=300, bbox='tight', format='png')
        plt.clf()


