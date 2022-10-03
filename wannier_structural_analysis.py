import numpy as np
from tqdm import tqdm
from read_trajectory import Trajectory
from progress_bar import ShowBar, ProgressBar
from periodic_boundary_condition import displacement, angle
from bond_asymmetry import BondAsymmetry
from charge_analysis import ChargeAnalysis
from matplotlib import pyplot as plt
from elemental_data import atomic_no, atomic_symbol
import matplotlib_style
from inputValues import (atom_name_2, atom_name_1, atom_name_3, 
                        rcut_tolerance_distance_selection, rcut_HostAtom_Wannier, 
                        fileTraj, SKIP, RESOLUTION, AnlgeCut_HostWannier_Host_SecondaryAtom, 
                        fileCharge, rcut_HostAtom_SecondaryAtom, Directory)

atom_name_1 = atomic_no(atom_name_1)
atom_name_2 = atomic_no(atom_name_2)
atom_name_3 = atomic_no(atom_name_3)

class WannierAnalysis(Trajectory):
    def __init__(self, filename=fileTraj, skip=SKIP, resolution = RESOLUTION):
        Trajectory.__init__(self, filename=filename, skip=skip, resolution = resolution)

    def compute_qnm(self, step, coord_atom1, coord_atom2):

        # Taking coord_atom2 (oxygen) as a center, computing other cation with in cutoff
        for s, atom1_other in enumerate(self.coordinates[step]):

            if (atom1_other[1:] != coord_atom1).any():

                if atom1_other[0] == atomic_no('Te') : #or atom1_other[0] == 'Tl':   # atom_name_1 ( for Te - O - Te)

                    coord_other_cation = np.array(atom1_other[1:], dtype=np.float_)

                    _, dist_other_12 = displacement(coord_other_cation, coord_atom2) 

                    if atom1_other[0] == atomic_no('Te'):
                        #rcut_HostAtom_SecondaryAtom = rcut_Te_O
                        pass
                    elif atom1_other[0] == atomic_no('Tl'):
                        raise TypeError ('Tl forms ionic bond, this method does not work here')
                        #rcut_HostAtom_SecondaryAtom = rcut_Tl_O

                    if dist_other_12 < rcut_HostAtom_SecondaryAtom:

                        for k, atom_wannier in enumerate(self.coordinates[step]):
                            if atom_wannier[0] == atom_name_2:
                                    coord_atom_wannier_defining_other_12 = np.array(atom_wannier[1:], dtype=np.float_)

                                    _, dist_2W_other_12 = displacement(coord_atom2, coord_atom_wannier_defining_other_12)
                                    _, dist_1W_other_12 = displacement(coord_other_cation, coord_atom_wannier_defining_other_12)

                                    distance_selection_other_12 = abs(dist_1W_other_12 + dist_2W_other_12 - dist_other_12)

                                    if distance_selection_other_12 < rcut_tolerance_distance_selection and dist_2W_other_12 < dist_1W_other_12:

                                        dist_1W_near_other_cation, coord_wannier_near_other_cation, count_number_of_wanniers_near_other_host = \
                                                self.get_farthest_atom(atom_name_search=atom_name_2, coord_ref=coord_other_cation,
                                                                  rcut=rcut_HostAtom_Wannier, step=step, Atom_ID="")

                                        degree_angle_other_12 = angle(coord_wannier_near_other_cation, coord_other_cation, coord_atom2)        

                                        # This will help in identifying a chemical bond
                                        if degree_angle_other_12 >= AnlgeCut_HostWannier_Host_SecondaryAtom:
                                            return 1 # Bonding secondary atom
                            
                        #return 0 # Atom found but couldn't satisfy the constraints

        return 0 # Isolated oxygen atom (couldn't find any other cation)

    def compute_neighbour_wannier_host_cation(self, filename="", compute_qnm_statistics=False, print_BO_NBO=False, 
                                              chargeAnalysis=False, method='DDEC6',write_output=False, 
                                              print_output=False, plot_wannier_dist=False,
                                              print_degeneracy=False):
        
            ''' 
            Function counts number of wannier centres around host atoms.
            atom_name_1 and atom_name_2 refers to host atom and wannier centers
            '''       
            Host_atom_coordination_std = [] # For calculating std for each fold
            Host_atom_coordination = np.zeros(30)
            Host_atom_coordination_qnm = np.zeros((30,30))
            
            angles = []
            Coordination = []

            if print_BO_NBO:
                BO = open('BO.dat', 'w')
                NBO = open('NBO.dat', 'w')
                print('You have chosen to write BO.dat and NBO.dat, consider disbaling Te ignoring option with different that one wannier (lone pair)')

            # For reading charge analysis file
            if chargeAnalysis:
                try:
                    with open(fileCharge, 'r') as openFileCharge:
                        self.dataCharge = openFileCharge.readlines()
                except:
                    raise f'{fileCharge} does not exist.' 

            if write_output:
                fw = open(Directory + atomic_symbol(atom_name_1) +'-Structure-analysis-result.dat','w')

                                                
            for step in tqdm(range(self.n_steps)):
               
                Host_atom_coordination_snap = np.zeros(30)
                #########################################################################
                # Printing live stepcount and total stepcound
                #if not print_output and not print_degeneracy:
                # print(f"Total Steps : {self.n_steps} ", end='\r')  
                # progressBar = "\rProgress: " + ProgressBar(self.n_steps -1 , step, 100)
                # ShowBar(progressBar)
                ########################################################################

                # For Charge Analysis
                if chargeAnalysis:
                    charge = ChargeAnalysis.chargeAnalysis(self, self.dataCharge, step, method=method)            
                
                # Counting number of host atoms
                count_number_of_host_atoms    =  0
                Atom_ID_Bonding_Secondary     = []
                Atom_ID_Non_Bonding_Secondary = []
                
                for atom_ID, atom1 in enumerate(self.coordinates[step]):
                    
                    if atom1[0] == atom_name_1:
                        coord_atom1 = np.array(atom1[1:], dtype=np.float_)
                        
                           
                        dist_atom_wannier_near_host, coord_atom_wannier_near_host, count_number_of_wanniers_near_host = \
                            self.get_farthest_atom(atom_name_search=atom_name_2, coord_ref=coord_atom1, 
                                              rcut=rcut_HostAtom_Wannier, step=step, Atom_ID=atom_ID,
                                              print_degeneracy=print_degeneracy)
                        
                        if atom_name_1 == atomic_no('Te') and count_number_of_wanniers_near_host != 1 :
                            print(f'WARNING: Te {atom_ID} encountered {count_number_of_wanniers_near_host} WFCs in step {step}')
                            continue
                        
                        # Not evaluated directly because sometime I use constraint, so for
                        # correct normalisation I count for those follow the constraint
                        count_number_of_host_atoms += 1
                        
                        # Looking for other oxygen atoms within cutoff radius
                        count_number_of_secondary_atoms = 0
                        count_number_of_secondary_bonding_atoms = 0         
                        
                        Atom_ID_secondary_list = []
                        Atom_ID_secondary_coordinates = []
                        
                        for atom_ID2, atom2 in enumerate(self.coordinates[step]):
                            if atom2[0] == atom_name_3:
                                coord_atom2 = np.array(atom2[1:], dtype=np.float_)
                                
                                _, dist_12 = displacement(coord_atom1,coord_atom2)
                                if dist_12 < rcut_HostAtom_SecondaryAtom: # cutoff for Te/Tl-O

                                    count_of_bonding_wannier = 0
                                    for l, atom_W in enumerate(self.coordinates[step]):
                                        
                                        if atom_W[0] == atom_name_2:
                                            coord_W = np.array(atom_W[1:], dtype=np.float_)
                                            
                                            _, dist_1W = displacement(coord_atom1, coord_W)
                                            _, dist_2W = displacement(coord_atom2, coord_W)
                                            
                                            if dist_2W < dist_1W and dist_2W < dist_12 and dist_1W < dist_12:  # Bonding wannier is close to anion than cation
                                            
                                                # Selecting Wannier centers which lies close to Te-O bond.
                                                dist_selection = abs(dist_12 - dist_1W - dist_2W)
                                                
                                                #if dist_2W < 1.0: # Second minima of O-X dist
                                                #    print(dist_selection)
                                                #    continue

                                                if dist_selection <= rcut_tolerance_distance_selection: # rcut for dist. tolerance
                                                    count_of_bonding_wannier += 1
                                                    
                                                    if count_of_bonding_wannier > 1 and atom_name_1 == atomic_no('Te'):
                                                        raise RuntimeError (f'WARNING: More than one bonding wannier found between {atomic_symbol(atom1[0])} {atom_ID} and '+\
                                                        f'{atomic_symbol(atom2[0])} {atom_ID2} with in step {step} with chosen (rcut_tolerance_distance_selection) parameter')
                                                    degree_angle = angle(coord_atom_wannier_near_host, coord_atom1, coord_atom2)
                                                    angles.append(degree_angle)
                                                    
                                                    # This will help in identifying a chemical bond
                                                    if degree_angle >= AnlgeCut_HostWannier_Host_SecondaryAtom:
                                                        count_number_of_secondary_atoms += 1
                                                        Atom_ID_secondary_list.append(atom_ID2)
                                                        Atom_ID_secondary_coordinates.append(coord_atom2)
                                                
                                                        if compute_qnm_statistics:
                                                            flag_bonding_secondary_atom = self.compute_qnm(step, coord_atom1, coord_atom2)
                                                            count_number_of_secondary_bonding_atoms += flag_bonding_secondary_atom
                                                            
                                                            if flag_bonding_secondary_atom == 0:
                                                                Atom_ID_Non_Bonding_Secondary.append(atom_ID2)
                                                                if print_BO_NBO:
                                                                    NBO.write(f'Step: {step}  Host-AtomID: {atom_ID}   NBO-ID: {atom_ID2} Distance: {dist_12} \n')
                                                            else:
                                                                Atom_ID_Bonding_Secondary.append(atom_ID2)
                                                                if print_BO_NBO:
                                                                    BO.write(f'Step: {step} Host-AtomID: {atom_ID}  BO-ID: {atom_ID2}  Distance: {dist_12} \n')
                

                        Host_atom_coordination[count_number_of_secondary_atoms] += 1
                        Host_atom_coordination_snap[count_number_of_secondary_atoms] += 1  # Test block for getting l-fold of each step
                        Host_atom_coordination_qnm[count_number_of_secondary_atoms,count_number_of_secondary_bonding_atoms] += 1
                                                            
                        if count_number_of_wanniers_near_host < 15:
                            BondAsymmetryValue = BondAsymmetry(count_number_of_secondary_atoms, coord_atom1, np.array(Atom_ID_secondary_coordinates))
                            if print_output:
                                if chargeAnalysis:
                                    print(f'Step: {step:3d} \t Atom-ID: {atom_ID:3d} \t Coordination: {count_number_of_secondary_atoms} \t' +
                                          f'Number-of-Wanniers: {count_number_of_wanniers_near_host} \t Charge: {round(charge[atom_ID],4):.4f} \t' +
                                          f'Bonding-Atom-ID: {str(Atom_ID_secondary_list):25} \t Asymmetry: {str(BondAsymmetryValue):8} \t  XYZ: {coord_atom1}')
                                else:
                                    print(f'Step: {step:3d} \t Atom-ID: {atom_ID:3d} \t Coordination: {count_number_of_secondary_atoms} \t' +
                                          f'Number-of-Wanniers: {count_number_of_wanniers_near_host} \t Bonding-Atom-ID: {str(Atom_ID_secondary_list):25} \t' +
                                          f'Asymmetry: {str(BondAsymmetryValue):8} \t XYZ: {coord_atom1}')
                                
                            if write_output:
                                if chargeAnalysis:
                                    fw.write(f'Step: {step:3d} \t Atom-ID: {atom_ID:3d} \t Coordination: {count_number_of_secondary_atoms} \t' +
                                             f'Number-of-Wanniers: {count_number_of_wanniers_near_host} \t Charge: {round(charge[atom_ID],4):.4f} \t' +
                                             f'Bonding-Atom-ID: {str(Atom_ID_secondary_list):25} \t Asymmetry: {str(BondAsymmetryValue):8} \t  XYZ: {coord_atom1} \n')
                                else: 
                                    fw.write(f'Step: {step:3d} \t Atom-ID: {atom_ID:3d} \t Coordination: {count_number_of_secondary_atoms} \t' +
                                             f'Number-of-Wanniers: {count_number_of_wanniers_near_host} \t Bonding-Atom-ID: {str(Atom_ID_secondary_list):25} \t'+
                                             f'Asymmetry: {str(BondAsymmetryValue):8} \t XYZ: {coord_atom1} \n')        
           

                Host_atom_coordination_std.append(Host_atom_coordination_snap)
                
                coord_snap = 0
                for index, values in enumerate(Host_atom_coordination_snap):
                   coord_snap += index * values
                
                coord_snap /= count_number_of_host_atoms
                Coordination.append(coord_snap)
            Host_atom_coordination_std = np.std(Host_atom_coordination_std, axis=0)
            
            if compute_qnm_statistics:
                print()
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                print('----------------------------------------')
                print('N_fold, Bridging anion, Count,  Percentage')
                print('----------------------------------------')
                total_percentage = 0
                for i, row in enumerate(Host_atom_coordination_qnm):
                    for j, col in enumerate(row):
                        if col != 0:
                            percentage = col/(count_number_of_host_atoms * self.n_steps) * 100
                            print(f'{i}        {j}         {round(col/self.n_steps,3):>8}    {round(percentage,2):>8}%')
                            total_percentage += percentage
                    if row.any() > 0:
                        print('----------------------------------------')
                print(f'Total {atomic_symbol(atom_name_1)} counted is {total_percentage:.2f}%')
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                
            # Printing the coordiantion number of nfold 
            Total_Percentage_N_folds = 0
            print()
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print('------------------------------------------------')
            print('N_fold, count of N_fold, Percentage, Std. dev.')
            print('------------------------------------------------')
            for n_fold, value in enumerate(Host_atom_coordination):
                if value != 0 :
                    percentage = value/(count_number_of_host_atoms * self.n_steps) * 100
                    percentage_std = Host_atom_coordination_std[n_fold] * (100/count_number_of_host_atoms)
                    print(f'{n_fold}         {value/self.n_steps:>8.3f}     {percentage:>8.2f} %   {"± "+str(round(percentage_std, 2)):>8} %')
                    Total_Percentage_N_folds += percentage
                    #Coordination_host_atom += n_fold * value 
            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            print(f'Total {atomic_symbol(atom_name_1)} counted is {Total_Percentage_N_folds:>4.2f} %  '+\
                  f'and total average coordination number is {np.mean(Coordination):>6.4f} ± {np.std(Coordination):<6.4f}. ')
            

            if write_output:
                if compute_qnm_statistics:
                    # Writing data to a file
                    fw.write('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
                    fw.write(f'N_fold, Bridging anion, Count, Percentage \n')
                    fw.write('----------------------------------------\n')
                    total_percentage = 0
                    for i, row in enumerate(Host_atom_coordination_qnm):
                        for j, col in enumerate(row):
                            if col != 0:
                                percentage = col/(count_number_of_host_atoms * self.n_steps) * 100
                                fw.write(f'{i}        {j}         {col/self.n_steps:>8.2f}    {percentage:>8.2f} \n')
                                total_percentage += percentage
                        if row.any() > 0:
                            fw.write('----------------------------------------\n')
                    fw.write('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
                    fw.write(f'Total {atomic_symbol(atom_name_1)} counted is {total_percentage:.2f}% \n')

                # Printing the coordiantion number of nfold 
                Total_Percentage_N_folds = 0
                #Coordination_host_atom = 0
                fw.write('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
                fw.write('N_fold, count of N_fold, Percentage,  Std. dev.   \n')
                fw.write('------------------------------------------------\n')
                for n_fold, value in enumerate(Host_atom_coordination):
                    if value != 0 :
                        percentage = value/(count_number_of_host_atoms * self.n_steps) * 100
                        percentage_std = Host_atom_coordination_std[n_fold] * (100/count_number_of_host_atoms)
                        fw.write(f'{n_fold}         {value/self.n_steps:>8.3f}     {percentage:>8.2f} %   {"± "+str(round(percentage_std, 2)):>8} % \n')
                        Total_Percentage_N_folds += percentage
                        #Coordination_host_atom += n_fold * value 
                fw.write('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n \n')
                fw.write(f'Total {atomic_symbol(atom_name_1)} counted is {Total_Percentage_N_folds:>4.2f} %  '+\
                         f'and total average coordination number is {np.mean(Coordination):>6.4f} ± {np.std(Coordination):<6.4f}. \n')
                fw.close()
                
            return np.mean(Coordination)


    def compute_neighbour_wannier_host_anion(self, filename="", 
                                             chargeAnalysis=False, method='DDEC6', 
                                             write_output=False, print_output=False,
                                             print_degeneracy=False):
        
        ''' 
        Function counts number of wannier centres around host atoms.
        This function is useful when host atom is anion and secondary atom 
        is cation.
        atom_name_1 and atom_name_2 refers to host atom and wannier centers
        '''
       
        Host_atom_coordination_std = [] # For calculating std for each fold
        Host_atom_coordination = np.zeros(30)
        Coordination = []

        if chargeAnalysis:
            try:
                with open(fileCharge, 'r') as openFileCharge:
                    self.dataCharge = openFileCharge.readlines()
            except:
                raise f'{fileCharge} does not exist.'  
                
        if write_output:
            fw = open(Directory + atomic_symbol(atom_name_1) +'-Structure-analysis-result.dat','w')
        

        for step in tqdm(range(self.n_steps)):
            
            # Printing live stepcount and total stepcound
            # print(f"Total Steps : {self.n_steps} ", end='\r')  
            #if not print_output:
            # progressBar = "\rProgress: " + ProgressBar(self.n_steps -1 , step, 100)
            # ShowBar(progressBar)

            # For charge Analysis
            if chargeAnalysis:
                charge = ChargeAnalysis.chargeAnalysis(self, self.dataCharge, step, method=method)


            # Counting number of host atoms
            count_number_of_host_atoms = 0
            
            Host_atom_coordination_snap = np.zeros(30)

            for atom_ID, value in enumerate(self.coordinates[step]):
                if value[0] == atom_name_1:
                    print()
                    print(charge[atom_ID], end=' ')
                    count_number_of_host_atoms += 1
                    coord_atom1 = np.array(value[1:], dtype=np.float_)
                    
                    # counter for  finding number of cation atoms attached
                    count_number_of_secondary_atoms = 0
                    
                    Atom_ID_secondary_list = []
                    Atom_ID_secondary_coordinates = []
                    
                    count_number_of_wannier_near_host_atom = 0

                    for atom_ID_W_lp, value_W_lp in enumerate(self.coordinates[step]):

                        if value_W_lp[0] == atom_name_2:
                            coord_wannier_near_host = np.array(value_W_lp[1:], dtype=np.float_)
                            _, dist_1W_near_host = displacement(coord_atom1, coord_wannier_near_host)
                            
                            if dist_1W_near_host <  rcut_HostAtom_Wannier:
                                count_number_of_wannier_near_host_atom += 1
                        

                    # Looking for secondary cation atom 
                    for atom_ID2, value2 in enumerate(self.coordinates[step]):
                        
                        
                        if value2[0] == atom_name_3 or value2[0] == atomic_no('Tl'): 
                            coord_atom2 = np.array(value2[1:], dtype=np.float_)
                            
                            # distance between host(primary) atom and secondary cation atom
                            _, dist_12 = displacement(coord_atom2, coord_atom1)
                            
                            ################################################################
                            ##################### Testt Block  ##############################
                            ################################################################
                            if value2[0] == atomic_no('Tl') and dist_12 <= 3.26:
                                print('Tl', end=' ')
                                continue
                            ################################################################
                            #################### End Test Block  ###########################
                            ################################################################

                            if dist_12 < rcut_HostAtom_SecondaryAtom : # cutoff for cation-anion
                            
                                for atom_ID_wannier, value_wannier in enumerate(self.coordinates[step]):
                                    if value_wannier[0] == atom_name_2:
                                        coord_atom_wannier = np.array(value_wannier[1:], dtype=np.float_)
                                    
                                        _, dist_1W = displacement(coord_atom1, coord_atom_wannier)
                                        _, dist_2W = displacement(coord_atom2, coord_atom_wannier)
                                        
                                        distance_selection = abs(dist_1W + dist_2W - dist_12)
                                        # if dist_1W < dist_12 and dist_2W < dist_12 and dist_1W > dist_2W and dist_2W < 0.5 and Atom_ID != 5:
                                        #    print(distance_selection, Atom_ID, Atom_ID_secondary, dist_1W, dist_2W, dist_12)  
                                        # continue
                                        

                                        if distance_selection < rcut_tolerance_distance_selection and dist_1W < dist_2W:
                                                                                        # Counting number of wannier centers near cation atom in cut-off radius
                                            
                                            dist_2W_near_cation, coord_wannier_near_cation, count_number_of_wanniers_near_cation = \
                                                  self.get_farthest_atom(atom_name_search=atom_name_2, coord_ref=coord_atom2,
                                                                    rcut=rcut_HostAtom_Wannier, step=step, Atom_ID=atom_ID,
                                                                    print_degeneracy=print_degeneracy)
                                            
                                            degree_angle = angle(coord_wannier_near_cation, coord_atom2, coord_atom1)
                                                            
                                            # This will help in identifying a chemical bond
                                            if degree_angle >= AnlgeCut_HostWannier_Host_SecondaryAtom:
                                                count_number_of_secondary_atoms += 1
                                                Atom_ID_secondary_list.append(atom_ID2)
                                                Atom_ID_secondary_coordinates.append(coord_atom2)
                                                print('Te', end=' ') 

                    #if count_number_of_wanniers_near_cation < 15:
                    BondAsymmetryValue = 'None' # BondAsymmetry(count_number_of_secondary_atoms, coord_atom1, Atom_ID_secondary_coordinates)
                    if print_output:
                        if chargeAnalysis:
                            print(f'Step: {step:3d} \t Atom-ID: {atom_ID:3d} \t Coordination: {count_number_of_secondary_atoms} \t' +
                                  f'Number-of-wanniers: {count_number_of_wannier_near_host_atom:2d} \t Charge: {round(charge[atom_ID],4):.4f} \t' + 
                                  f'Bonding-Atom-ID: {str(Atom_ID_secondary_list):15} \t Asymmetry: {str(BondAsymmetryValue):8} \t XYZ: {coord_atom1}')
                        else: 
                            print(f'Step: {step:3d} \t Atom-ID: {atom_ID:3d} \t Coordination: {count_number_of_secondary_atoms} \t' +
                                  f'Number-of-wanniers: {count_number_of_wannier_near_host_atom:2d} \t Bonding-Atom-ID: {str(Atom_ID_secondary_list):15} \t'+
                                  f'Asymmetry: {str(BondAsymmetryValue):8} \t XYZ: {coord_atom1}')
                        
                    if write_output:
                        if chargeAnalysis:
                            fw.write(f'Step: {step} \t Atom-ID: {atom_ID:3d} \t Coordination: {count_number_of_secondary_atoms} \t'+
                                     f'Number-of-wanniers: {count_number_of_wannier_near_host_atom:2d} \t Charge: {round(charge[atom_ID],4):.4f} \t' +
                                     f'Bonding-Atom-ID: {str(Atom_ID_secondary_list):15} \t Asymmetry: {str(BondAsymmetryValue):8} \t XYZ: {coord_atom1} \n')
                        else:
                            fw.write(f'Step: {step} \t Atom-ID: {atom_ID:3d} \t Coordination: {count_number_of_secondary_atoms} \t' +
                                     f'Number-of-wanniers: {count_number_of_wannier_near_host_atom:2d} \t Bonding-Atom-ID: {str(Atom_ID_secondary_list):15}, \t' +
                                     f'Asymmetry: {str(BondAsymmetryValue):8} \t XYZ: {coord_atom1} \n')
                                 
                    Host_atom_coordination[count_number_of_secondary_atoms] += 1
                    Host_atom_coordination_snap[count_number_of_secondary_atoms] += 1
                
            Host_atom_coordination_std.append(Host_atom_coordination_snap)
            coord_snap = 0
            for index, values in enumerate(Host_atom_coordination_snap):
               coord_snap += index * values

            coord_snap /= count_number_of_host_atoms
            Coordination.append(coord_snap)
            

        Host_atom_coordination_std = np.std(Host_atom_coordination_std, axis=0)

        # Printing the coordiantion number of nfold 
        Total_Percentage_N_folds = 0
        #Coordination_host_atom = 0
        print()
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('------------------------------------------------')
        print('N_fold, count of N_fold, Percentage, Std. dev.')
        print('------------------------------------------------')
        for n_fold, value in enumerate(Host_atom_coordination):
            if value != 0 :
                percentage = value/(count_number_of_host_atoms * self.n_steps) * 100
                percentage_std = Host_atom_coordination_std[n_fold] * (100/count_number_of_host_atoms)
                print(f'{n_fold}         {round(value/self.n_steps,3):>8}     {round(percentage,2):>8} %   {"± "+str(round(percentage_std, 2)):>8} %')
                Total_Percentage_N_folds += percentage
                #Coordination_host_atom += n_fold * value 
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print()
        print(f'Total {atomic_symbol(atom_name_1)} counted is {round(Total_Percentage_N_folds, 2)}%'+\
              f'and total average coordination number is {round(np.mean(Coordination),4)} ± {round(np.std(Coordination), 4)}.')
        #print(f'Total {atomic_symbol(atom_name_1)} counted is {Total_Percentage_N_folds}% and total average coordination number is {Coordination_host_atom/(count_number_of_host_atoms * self.n_steps)}.')
                                
        if write_output:
            Total_Percentage_N_folds = 0
            Coordination_host_atom = 0
            fw.write('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
            fw.write('N_fold, count of N_fold, Percentage,  Std. dev.   \n')
            fw.write('------------------------------------------------\n')
            for n_fold, value in enumerate(Host_atom_coordination):
                if value != 0 :
                    percentage = value/(count_number_of_host_atoms * self.n_steps) * 100
                    percentage_std = Host_atom_coordination_std[n_fold] * (100/count_number_of_host_atoms)
                    fw.write(f'{n_fold}         {round(value/self.n_steps,3):>8}     {round(percentage,2):>8} %   {"± "+str(round(percentage_std, 2)):>8} % \n')
                    Total_Percentage_N_folds += percentage
                    #Coordination_host_atom += n_fold * value 
            fw.write('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n \n')
            fw.write(f'Total {atomic_symbol(atom_name_1)} counted is {round(Total_Percentage_N_folds, 2)}%'+\
                     f'and total average coordination number is {round(np.mean(Coordination),4)} ± {round(np.std(Coordination), 4)}. \n')
            fw.close()

        return None


    def get_farthest_atom(self, 
                          atom_name_search, 
                          coord_ref, 
                          rcut: float, 
                          step: int, 
                          Atom_ID: int, 
                          print_degeneracy=False):
        '''
            Looks for the atom located farthest from some reference coordinate

            Parameters : 
            -----------------------------------------------------------------
            atom_name_search : Name of the atom to look for located farthest in rcut radius.
            coord_ref : Coordinate of the reference atom
            rcut : Radius of the cut-off region to search for the atom.
            step : count of the n-th frame
            Atom_ID : Atom ID of the reference atom
            print_degeneracy : To print warning if non-expected atoms are found

            Return :
            -----------------------------------------------------------------
            distance : Distance of the atom located farthest from the reference atom.
            coord    : Coordinate of the atom located farthest.
            count    : Number of such species of atom in rcut region.

        '''
        count_number_of_wanniers_near_host = 0

        coord_of_various_wannier_centers_near_host_atom = []
        dist_of_various_wannier_centers_near_host_atom  = []
        Atom_ID_Wannier_list = []
        
        for atom_ID_wannier, atom_wannier in enumerate(self.coordinates[step]):

            if atom_wannier[0] == atom_name_search:
                _, dist_atom_wannier_near_host = displacement(coord_ref, atom_wannier[1:])

                if dist_atom_wannier_near_host < rcut :   # Cutoff for Te-W 
                    count_number_of_wanniers_near_host += 1
                    coord_of_various_wannier_centers_near_host_atom.append(atom_wannier[1:])
                    dist_of_various_wannier_centers_near_host_atom.append(dist_atom_wannier_near_host)

        if count_number_of_wanniers_near_host > 1 :
            if print_degeneracy:
                for i, value in enumerate(Atom_ID_Wannier_list):
                    print(f'WARNING: DEGENERACY {count_number_of_wanniers_near_host} -- Step: {step} \t ' +
                          f'Atom-ID: {Atom_ID} \t Atom-ID-Wannier: {value} \t ' +
                          f'Distance-Atom-Wannier: {dist_of_various_wannier_centers_near_host_atom[i]:8.6f}') 
                #print()
            coord_atom_wannier_near_host = \
            coord_of_various_wannier_centers_near_host_atom[
                dist_of_various_wannier_centers_near_host_atom.index(max(\
                    dist_of_various_wannier_centers_near_host_atom))]
            dist_atom_wannier_near_host = max(dist_of_various_wannier_centers_near_host_atom)
        elif count_number_of_wanniers_near_host == 1:
            coord_atom_wannier_near_host = coord_of_various_wannier_centers_near_host_atom[0]
            dist_atom_wannier_near_host = dist_of_various_wannier_centers_near_host_atom[0]
        else:
            raise RuntimeError(f'WARNING: There are no wanniers center near host atom with Atom ID {Atom_ID} in step {step}. '+ 
                               f'Check rcut_HostAtom_Wannier/Lattice parameters in inputValues.py')

        return dist_atom_wannier_near_host, coord_atom_wannier_near_host, count_number_of_wanniers_near_host
