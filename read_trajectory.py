import numpy as np
import json
from elemental_data import atomic_no
from inputValues import atom_name_2, atom_name_1, atom_name_3

class Trajectory:
    
    '''
    atom_name_1, atom_name_2, atom_name_3 should be defined in the code. For testig purpose.

    Input parameters :
        Filename, Skip : This works as jumping, Resolution : To tune the grid in calculation of RDF
   
    Output parameters :

        self.coordinates: This is a three dimensional matrix with rows as atomic cooridnate, 
        column the element symbol and three successive directions (x, y, z) and hieght is the
        step number in the trajectory file.
    '''

    def __init__(self,filename, skip=1, resolution = 500):
        self.filename = filename
        self.skip = skip
        self.resolution = resolution
        self.parse_input()
        

    def parse_input(self):  # for getting coordinates from the trajectory file
        global Atom_type
        with open(self.filename, 'r') as fileOpen: 
            data = fileOpen.readlines()  

        # First line of XYZ file (Number of atoms)
        self.n_atoms = int(data[0].split()[0]) 
        
        # Total number of configurations 
        self.n_steps_total = int(len(data) / (self.n_atoms +2))  
        
        # To skip steps (skip = 1 refers to no skip)
        self.n_steps = self.n_steps_total // self.skip 

        # Getting types of atoms 
        self.atom_list = [line.split()[0] for line in data[2:self.n_atoms + 2]]  

        self.Atom_type = list(dict.fromkeys(self.atom_list))

        # print(self.atom_name_1, self.atom_name_2, self.atom_name_3)

        if atom_name_1 not in self.Atom_type or atom_name_2 not in self.Atom_type or atom_name_3 not in self.Atom_type:
            raise ValueError('Element not found in the coordinate file')

        self.number_of_atoms = len(self.atom_list)  # This variable is create as in case of wannier calculation there are additional X atoms in cp2k output which gives irresonable number of atoms 
        # and cannot be used with bader analysis.            
            
        # This block removes the X atom in atom_list array to count proper number of atoms for bader analyses but it fails for PDF calculation becuase X is not there   
        try:
            if atom_name_2 == 'X' :
                Total_number_X_atom = self.atom_list.count(atom_name_2)
                self.number_of_atoms = self.number_of_atoms - Total_number_X_atom
        except ValueError:
            pass
            
            
        # Generating n_steps 2D matrices with n_atoms (rows) and 4 columns
        self.coordinates = np.zeros((self.n_steps, self.n_atoms, 4), dtype=np.float_) 
        self.coordinates_energy = np.zeros((self.n_steps, self.n_atoms, 1), dtype=np.float_)

        # Iterating over each considered configuration after skipping
        for step in range(self.n_steps): 
            coords = np.zeros((self.n_atoms, 4), dtype= object)
            coords_energy = np.zeros((self.n_atoms, 1), dtype = 'float')
            
            i = step * self.skip * (self.n_atoms + 2)  

            # Collecting coordinate data in each configuration
            for row, row_values in enumerate(data[i + 2 : i + self.n_atoms + 2]): 
                coords[row, 0] = atomic_no(row_values.split()[0])
                coords[row, 1:] = [float(value) for value in row_values.split()[1:4]]
                
                # here let's assume, 5th column contains the energy, will mainly useful for 
                # calculating the energies of wannier center. This require some preprocessing.
                if len(row_values.split()) == 5:
                    coords_energy[row] = float(row_values.split()[4])


            # Storing cooridnate values of considered configuration
            self.coordinates[step] = coords  
            self.coordinates_energy[step] = coords_energy
