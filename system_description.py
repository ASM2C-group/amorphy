from read_trajectory import Trajectory
from inputValues import fileTraj, SKIP, RESOLUTION, A, B, C, ALPHA, BETA, GAMMA, Directory
from collections import Counter
from elemental_data import atomic_mass 
import datetime

class SystemInfo(Trajectory):
    def __init__(self, filename=fileTraj, skip=SKIP, resolution = RESOLUTION):
        Trajectory.__init__(self, filename=filename, skip=skip, resolution = resolution)
        self.volume()
        self.molar_mass()
        self.molar_density()
        self.number_density()
        self.atom_info()
        self.write_info()
        self.additional_info()

    def volume(self):
        self.volume = A * B * C

    
    def molar_mass(self):
        elements = Counter(self.atom_list)
        mol_mass = 0
        for value in elements.items():
            mol_mass += atomic_mass(value[0], gram=True)*value[1]
        self.mol_mass = mol_mass

    def molar_density(self):
        self.mol_density = self.mol_mass / self.volume        

    def number_density(self):
        self.num_density = self.number_of_atoms/self.volume


    def atom_info(self):
        self.elements = Counter(self.atom_list)
   
    def write_info(self):
        openfile = (Directory+'System-info.out')
        with open(openfile, 'w') as fw:
            fw.write(f'{str(" #######  Date/Time"):>25} : {datetime.datetime.now()}  ####### \n\n')
            fw.write(f'{str("Total number of steps"):<50} = {self.n_steps_total} \n')
            fw.write(f'{str("Number of steps considered after jump of step")} {SKIP:<4} = {self.n_steps} \n \n')
            fw.write(f'{str("Types of atoms"):<50} = {self.Atom_type} \n \n')
            fw.write(f'{str("Atom Info"):<50} \n')
            for value in self.elements.items():
                fw.write(f'{str("Number of"):>41}  {value[0]:>2} atom = {value[1]} \n')
            fw.write(f'\n{str("Lattice parameters"):<48} A = {A:>6f} Å \n {str("B"):>49} = {B:>6f} Å \n {str("C"):>49} = {C:6f} Å \n {str("ALPHA"):>49} = {ALPHA} deg \n  {str("BETA"):>48} = {BETA} deg \n  {str("GAMMA"):>48} = {GAMMA} deg \n \n')
            fw.write(f'{str("Simulation cell volume"):<50} = {self.volume:4.4f} Å^3 \n')
            fw.write(f'{str("Molar density"):<50} = {self.mol_density:4.4f} g/cm^3 \n')
            fw.write(f'{str("Number density"):<50} = {self.num_density:4.4f} cm^-3 \n\n')

    def additional_info(self, info=False):
        openfile = (Directory+'System-info.out')
        with open(openfile, 'a') as fw:
            if info:
                fw.write(f'\n {info} \n')
