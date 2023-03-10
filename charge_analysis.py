from read_trajectory import Trajectory
from inputValues import fileTraj, SKIP, RESOLUTION
from periodic_boundary_condition import displacement
import numpy as np

eps = 10e-4

class ChargeAnalysis(Trajectory):
    def __init__(self, filename=fileTraj, skip=SKIP, resolution = RESOLUTION):
        Trajectory.__init__(self, filename=filename, skip=skip, resolution = resolution)
    
    def chargeAnalysis(self, fileCharge, step, method='DDEC6'):
            
            if method == 'Bader':
            
                ''' Here ACF file is read. The ACF file is generated by Bader charge analysis, code written by Henkelmen group.
                    For performing bader analysis, .cube file required which in return can be obtained from cp2k.
                    Appended ACF file calculated on various snapshots are used here to analyse the bader in corelation with the 
                    Wannier calcuations. 
                    CAREFUL : About the order of wannier snapshot and bader snapshots, both should be in proper order.
                '''

                Charges = []

                i = step * (self.number_of_atoms + 6)  # 6 entries are extra in bader file

                for value in self.dataCharge[i:i+self.number_of_atoms+6]:

                    if '#' in value.split()[0] or '-' in value.split()[0] or len(value.split()) < 7 : continue 

                    atomID = int(value.split()[0])
                    atomX  = float(value.split()[1])
                    atomY  = float(value.split()[2])
                    atomZ  = float(value.split()[3])
                    charge = float(value.split()[4])

                    # wrapped = wrap_positions([[atomX, atomY, atomZ]], cell=LatticeMatrix, pbc=True)

                    if self.coordinates[0][atomID-1,0] == 'Tl':
                        charge = 13 - charge
                    elif self.coordinates[0][atomID-1,0] == 'Te':
                        charge = 6 - charge 
                    elif self.coordinates[0][atomID-1,0] == 'O':
                        charge = 6 - charge 
                    else:
                        print(f'{self.coordinates[0][atomID-1,0]} See error')

                    # print(f'Atom ID : {atomID}  Atom Name: {self.coordinates[0][atomID-1,0]}  Charge : {round(charge,4)}')
                    # print(f'{atomID}  {round(wrapped[0][0],5):>8} {round(wrapped[0][1],5):>8} {round(wrapped[0][2],5):>8} {round(charge,4):>8}')
                    Charges.append(charge)

                return Charges
            
            elif method == 'DDEC6':
            
                Charges = []
                
                i = step * (self.number_of_atoms + 2)
               
                index = 0
                for index, value in enumerate(self.dataCharge[i+2:i+self.number_of_atoms+2]): 

                    atomName = str(value.split()[0])
                    atomCoord  = np.array(value.split()[1:4], dtype=np.float_)
                    charge = float(value.split()[4])

                    _, Dist = displacement(atomCoord, self.coordinates[step][index][1:])
                    
                    if Dist > eps:
                        raise RuntimeError('There is mismatch of coords between charge and traj file')
                    
                    Charges.append(charge)
            
                return Charges
