import numpy as np
from tqdm import tqdm
from read_trajectory import Trajectory
from progress_bar import ShowBar, ProgressBar
from periodic_boundary_condition import displacement, angle
from bond_asymmetry import BondAsymmetry
from charge_analysis import ChargeAnalysis
from matplotlib import pyplot as plt
import matplotlib_style
from inputValues import (atom_name_2, atom_name_1, atom_name_3, rcut_Te_O, rcut_Tl_O, 
                        rcut_tolerance_distance_selection, rcut_HostAtom_Wannier, 
                        fileTraj, SKIP, RESOLUTION, AnlgeCut_HostWannier_Host_SecondaryAtom, 
                        fileCharge, rcut_HostAtom_SecondaryAtom, Directory)
from topology import compute_coordination

class WannierAnalysis(Trajectory):
    def __init__(self, filename=fileTraj, skip=SKIP, resolution = RESOLUTION):
        Trajectory.__init__(self, filename=filename, skip=skip, resolution = resolution)

    def compute_qnm(self, step, coord_atom1, coord_atom2):

        # Taking coord_atom2 (oxygen) as a center, computing other cation with in cutoff
        for s, atom1_other in enumerate(self.coordinates[step]):

            if (atom1_other[1:] != coord_atom1).any():

                if atom1_other[0] == 'Te' : #or atom1_other[0] == 'Tl':   # atom_name_1 ( for Te - O - Te)

                    coord_other_cation = np.array(atom1_other[1:], dtype=np.float_)

                    _, dist_other_12 = displacement(coord_other_cation, coord_atom2) 

                    if atom1_other[0] == 'Te':
                        #rcut_HostAtom_SecondaryAtom = rcut_Te_O
                        pass
                    elif atom1_other[0] == 'Tl':
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

                                        dist_1W_near_other_cation, coord_wannier_near_other_cation, count_number_of_wanniers_near_other_host, _ = \
                                                self.get_farthest_atom(atom_name_search=atom_name_2, coord_ref=coord_other_cation,
                                                                  rcut=rcut_HostAtom_Wannier, step=step, Atom_ID="")

                                        degree_angle_other_12 = angle(coord_wannier_near_other_cation, coord_other_cation, coord_atom2)        

                                        # This will help in identifying a chemical bond
                                        if degree_angle_other_12 >= AnlgeCut_HostWannier_Host_SecondaryAtom:
                                            return 1 # Bonding secondary atom
                            
                        #return 0 # Atom found but couldn't satisfy the constraints

        return 0 # Isolated oxygen atom (couldn't find any other cation)

    def compute_neighbour_wannier(self, filename="", compute_qnm_statistics=False, print_BO_NBO=False, 
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
            max_dist = []
            Average_Tl_coordination = np.zeros(12)
            Tl_coordination_stats = []
            
            if print_BO_NBO:
                BO = open('BO.dat', 'w')
                NBO = open('NBO.dat', 'w')

            # For reading charge analysis file
            if chargeAnalysis:
                try:
                    with open(fileCharge, 'r') as openFileCharge:
                        self.dataCharge = openFileCharge.readlines()
                except:
                    raise f'{fileCharge} does not exist.'   
                                                
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
                count_number_of_host_atoms = 0
                Atom_ID_Bonding_Secondary = []
                Atom_ID_Non_Bonding_Secondary = []
                wannier_distance_energy = []
                
                for Atom_ID, atom1 in enumerate(self.coordinates[step]):
                    
                    if atom1[0] == atom_name_1:
                        #count_number_of_host_atoms += 1
                        coord_atom1 = np.array(atom1[1:], dtype=np.float_)
                        
                        # Atom_ID_secondary = 0 # Secondary Atom ID
                           
                        dist_atom_wannier_near_host, coord_atom_wannier_near_host, count_number_of_wanniers_near_host,  Wannier_Distance_Energy = \
                            self.get_farthest_atom(atom_name_search=atom_name_2, coord_ref=coord_atom1, 
                                              rcut=rcut_HostAtom_Wannier, step=step, Atom_ID=Atom_ID,
                                              print_degeneracy=print_degeneracy)
                        
                        wannier_distance_energy.extend(Wannier_Distance_Energy)
                        #if atom_name_1 == 'Tl' and count_number_of_wanniers_near_host == 5 :
                        #    print('Tl less than 6 wannier encountered', Atom_ID)
                        #    continue
                        if atom_name_1 == 'Te' and count_number_of_wanniers_near_host != 1 :
                            print('Te more than 1 wannier encountered')
                            continue
                        
                        # Not evaluated directly because sometime I use constraint, so for
                        # correct normalisation I count for those follow the constraint
                        count_number_of_host_atoms += 1
                        max_dist.append(dist_atom_wannier_near_host)
                        
                        # Looking for other oxygen atoms within cutoff radius
                        count_number_of_secondary_atoms = 0
                        count_number_of_secondary_bonding_atoms = 0         
                        
                        Atom_ID_secondary_list = []
                        Atom_ID_secondary_coordinates = []
                        
                        for Atom_ID_secondary, atom2 in enumerate(self.coordinates[step]):
                            
                            if atom2[0] == atom_name_3:
                                coord_atom2 = np.array(atom2[1:], dtype=np.float_)
                                
                                # distance between primary and secondary atoms
                                _, dist_12 = displacement(coord_atom1,coord_atom2)

                                                            
                                if dist_12 < rcut_HostAtom_SecondaryAtom: # cutoff for Te/Tl-O

                                    ## Test Block
                                    #_, dist_wannier_cation = displacement(coord_atom_wannier_near_host, coord_atom2)
                                    #print(abs(dist_atom_wannier_near_host+dist_wannier_cation-dist_12))    
                                    #continue
                                    
                                    for l, atom_W in enumerate(self.coordinates[step]):
                                        if atom_W[0] == atom_name_2:
                                            coord_W = np.array(atom_W[1:], dtype=np.float_)
                                            
                                            # distance between primary and Wannier
                                            _, dist_1W = displacement(coord_atom1, coord_W)
                                            # distance between secondary and Wannier
                                            _, dist_2W = displacement(coord_atom2, coord_W)
                                            
                                            if dist_2W < dist_1W:  # Bonding wannier is close to anion than cation
                                            
                                                # Selecting Wannier centers which lies close to Te-O bond.
                                                dist_selection = abs(dist_12 - dist_1W - dist_2W)

                                                if dist_selection <= rcut_tolerance_distance_selection: # rcut for dist. tolerance
                                                
                                                    degree_angle = angle(coord_atom_wannier_near_host, coord_atom1, coord_atom2)
                                                        
                                                    angles.append(degree_angle)
                                                    
                                                    # This will help in identifying a chemical bond
                                                    if degree_angle >= AnlgeCut_HostWannier_Host_SecondaryAtom:
                                                        count_number_of_secondary_atoms += 1
                                                        Atom_ID_secondary_list.append(Atom_ID_secondary)
                                                        Atom_ID_secondary_coordinates.append(coord_atom2)
                                                
                                                        if compute_qnm_statistics:
                                                            flag_bonding_secondary_atom = self.compute_qnm(step, coord_atom1, coord_atom2)
                                                            count_number_of_secondary_bonding_atoms += flag_bonding_secondary_atom
                                                            
                                                            if flag_bonding_secondary_atom == 0:
                                                                Atom_ID_Non_Bonding_Secondary.append(Atom_ID_secondary)
                                                                if print_BO_NBO:
                                                                    NBO.write(f'Step: {step}  Host-AtomID: {Atom_ID}   NBO-ID: {Atom_ID_secondary} Distance: {dist_12} \n')
                                                            else:
                                                                Atom_ID_Bonding_Secondary.append(Atom_ID_secondary)
                                                                if print_BO_NBO:
                                                                    BO.write(f'Step: {step} Host-AtomID: {Atom_ID}  BO-ID: {Atom_ID_secondary}  Distance: {dist_12} \n')
                

                        Host_atom_coordination[count_number_of_secondary_atoms] += 1
                        Host_atom_coordination_snap[count_number_of_secondary_atoms] += 1  # Test block for getting l-fold of each step
                        Host_atom_coordination_qnm[count_number_of_secondary_atoms,count_number_of_secondary_bonding_atoms] += 1
                                                            
                        if count_number_of_wanniers_near_host < 15:
                            BondAsymmetryValue = BondAsymmetry(count_number_of_secondary_atoms, coord_atom1, np.array(Atom_ID_secondary_coordinates))
                            if print_output:
                                if chargeAnalysis:
                                    print(f'Step: {step:3d} \t Atom-ID: {Atom_ID:3d} \t Coordination: {count_number_of_secondary_atoms} \t' +
                                        f'Number-of-Wanniers: {count_number_of_wanniers_near_host} \t Charge: {round(charge[Atom_ID-1],4):.4f} \t' +
                                        f'Bonding-Atom-ID: {Atom_ID_secondary_list} \t Asymmtery: {BondAsymmetryValue} \t  XYZ: {coord_atom1}')
                                else:
                                    print(f'Step: {step:3d} \t Atom-ID: {Atom_ID:3d} \t Coordination: {count_number_of_secondary_atoms} \t' +
                                        f'Number-of-Wanniers: {count_number_of_wanniers_near_host} \t Bonding-Atom-ID: {Atom_ID_secondary_list} \t' +
                                        f'Asymmtery: {BondAsymmetryValue} \t XYZ: {coord_atom1}')
                                
                            if write_output:
                                fw = open(Directory + atom_name_1 +'-Structure-analysis-result.dat','a')
                                if chargeAnalysis:
                                    fw.write(f'Step: {step:3d} \t Atom-ID: {Atom_ID:3d} \t Coordination: {count_number_of_secondary_atoms} \t' +
                                            f'Number-of-Wanniers: {count_number_of_wanniers_near_host} \t Charge: {round(charge[Atom_ID-1],4):.4f} \t' +
                                            f'Bonding Atom ID: {Atom_ID_secondary_list} \t Asymmtery: {BondAsymmetryValue} \t  XYZ: {coord_atom1} \n')
                                else: 
                                    fw.write(f'Step: {step:3d} \t Atom-ID: {Atom_ID:3d} \t Coordination: {count_number_of_secondary_atoms} \t' +
                                            f'Number-of-Wanniers: {count_number_of_wanniers_near_host} \t Bonding-Atom-ID: {Atom_ID_secondary_list} \t'+
                                            f'Asymmtery: {BondAsymmetryValue} \t XYZ: {coord_atom1} \n')        
           

                Host_atom_coordination_std.append(Host_atom_coordination_snap)
            
            Host_atom_coordination_std = np.std(Host_atom_coordination_std, axis=0)
            
            if compute_qnm_statistics:
                print()
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                print('----------------------------------------')
                print('N_fold, Bridging anion, Count, Percentage')
                print('----------------------------------------')
                total_percentage = 0
                for i, row in enumerate(Host_atom_coordination_qnm):
                    for j, col in enumerate(row):
                        if col != 0:
                            percentage = col/(count_number_of_host_atoms * self.n_steps) * 100
                            print(f'{i}        {j}         {round(col/self.n_steps,3):>8}    {round(percentage,2):>8}')
                            total_percentage += percentage
                    if row.any() > 0:
                        print('----------------------------------------')
                print(f'Total {atom_name_1} counted is {round(total_percentage,2)}%.')
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                
            # Printing the coordiantion number of nfold 
            Total_Percentage_N_folds = 0
            Coordination_host_atom = 0
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
                    Coordination_host_atom += n_fold * value 
            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            print()
            average_coordination =  Coordination_host_atom/(count_number_of_host_atoms * self.n_steps)
            print(f'Total {atom_name_1} counted is {round(Total_Percentage_N_folds, 2)}% and total average coordination number is {average_coordination}.')
            
            Wannier_Distance_Energy = np.array(wannier_distance_energy, dtype='float')
            Wannier_Distance_Energy = np.atleast_2d(Wannier_Distance_Energy)

            if plot_wannier_dist:
                plt.figure(figsize=(12,10))
                plt.hist(Wannier_Distance_Energy[:,0], bins=50)
                plt.xlabel('Distance', fontsize=15)
                plt.ylabel('count', fontsize=15)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlim(0,1)
                plt.title('Histogram of distances of Wannier centers to the host atom', fontsize=18)
                plt.show()

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
                                fw.write(f'{i}        {j}         {round(col/self.n_steps,2):>8}    {round(percentage,2):>8} \n')
                                total_percentage += percentage
                        if row.any() > 0:
                            fw.write('----------------------------------------\n')
                    fw.write('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
                    fw.write(f'Total {atom_name_1} counted is {round(total_percentage, 2)}%.\n')
                    

                # Printing the coordiantion number of nfold 
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
                        Coordination_host_atom += n_fold * value 
                fw.write('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n \n')
                fw.write(f'Total {atom_name_1} counted is {round(Total_Percentage_N_folds,2)}% and total average coordination number is {average_coordination}. \n')
                fw.close()
                
            return average_coordination, max_dist, angles, Wannier_Distance_Energy


    def compute_neighbour_wannier_host_anion(self, filename="", chargeAnalysis=False, method='DDEC6', 
                                                write_output=False, print_output=False, plot_wannier_dist=False,
                                                print_degeneracy=False):
        
        ''' 
        Function counts number of wannier centres around host atoms.
        This function is useful when host atom is anion and secondary atom 
        is cation.
        atom_name_1 and atom_name_2 refers to host atom and wannier centers
        '''
        
        Host_atom_coordination = np.zeros(30)
        
        if chargeAnalysis:
            try:
                with open(fileCharge, 'r') as openFileCharge:
                    self.dataCharge = openFileCharge.readlines()
            except:
                raise f'{fileCharge} does not exist.'  
                
        Wannier_Distance_Energy = []

        for step in tqdm(range(self.n_steps)):
            
            # Printing live stepcount and total stepcound
            # print(f"Total Steps : {self.n_steps} ", end='\r')  
            #if not print_output:
            # progressBar = "\rProgress: " + ProgressBar(self.n_steps -1 , step, 100)
            # ShowBar(progressBar)

            # For charge Analysis
            if chargeAnalysis:
                charge = self.chargeAnalysis(fileCharge, step, method=method)    

            # Counting number of host atoms
            count_number_of_host_atoms = 0
            Atom_ID = 0  # Primary Atom ID
            
            
            for i, atom1 in enumerate(self.coordinates[step]):
                
                # Serial number of atom in coordinate file
                Atom_ID += 1
                
                if atom1[0] == atom_name_1:
                    count_number_of_host_atoms += 1
                    coord_atom1 = np.array(atom1[1:], dtype=np.float_)
                    
                    # counter for  finding number of cation atoms attached
                    count_number_of_secondary_atoms = 0
                    
                    Atom_ID_secondary = 0 # Secondary Atom ID
                    Atom_ID_secondary_list = []
                    Atom_ID_secondary_coordinates = []
                    
                    Atom_ID_Wannier_host = 0
                    count_number_of_wannier_near_host_atom = 0

                    for l, atom_W in enumerate(self.coordinates[step]):
                        Atom_ID_Wannier_host += 1

                        if atom_W[0] == atom_name_2:
                            coord_wannier_near_host = np.array(atom_W[1:], dtype=np.float_)
                            _, dist_1W_near_host = displacement(coord_atom1, coord_wannier_near_host)
                            Wannier_Distance_Energy.append([dist_1W_near_host, self.coordinates_energy[step][Atom_ID_Wannier_host-1], Atom_ID])
                            # print(dist_1W_near_host, self.coordinates_energy[step][Atom_ID_Wannier_host-1]) ## Test
                            if dist_1W_near_host <  rcut_HostAtom_Wannier:
                                count_number_of_wannier_near_host_atom += 1
                        
                    
                    # Looking for secondary cation atom 
                    for j, atom2 in enumerate(self.coordinates[step]):
                        
                        Atom_ID_secondary += 1
                        
                        if atom2[0] == 'Tl' or atom2[0] == 'Te': # [if atom2[0] == atom_name_3] is not used since secondary atom can be any cation
                        # if atom2[0] == 'O': # TEST #
                            coord_atom2 = np.array(atom2[1:], dtype=np.float_)
                            
                            # distance between host(primary) atom and secondary cation atom
                            _, dist_12 = displacement(coord_atom2, coord_atom1)
                            
                            #if atom2[0] == 'Tl':
                            #    rcut_HostAtom_SecondaryAtom = rcut_Tl_O
                            #elif atom2[0] == 'Te':
                            #    rcut_HostAtom_SecondaryAtom = rcut_Te_O
                            
                            if dist_12 < rcut_HostAtom_SecondaryAtom : # cutoff for cation-anion
                            
                                for k, atom_wannier in enumerate(self.coordinates[step]):
                                    if atom_wannier[0] == atom_name_2:
                                        coord_atom_wannier = np.array(atom_wannier[1:], dtype=np.float_)
                                    
                                        _, dist_1W = displacement(coord_atom1, coord_atom_wannier)
                                        _, dist_2W = displacement(coord_atom2, coord_atom_wannier)
                                        
                                        distance_selection = abs(dist_1W + dist_2W - dist_12)
                                        # if dist_1W < dist_12 and dist_2W < dist_12 and dist_1W > dist_2W and dist_2W < 0.5 and Atom_ID != 5:
                                        #    print(distance_selection, Atom_ID, Atom_ID_secondary, dist_1W, dist_2W, dist_12)  
                                        # continue
                                        

                                        if distance_selection < rcut_tolerance_distance_selection and dist_1W < dist_2W:
                                                                                        # Counting number of wannier centers near cation atom in cut-off radius
                                            
                                            dist_2W_near_cation, coord_wannier_near_cation, count_number_of_wanniers_near_cation, Wannier_Distance_Energy = \
                                                  self.get_farthest_atom(atom_name_search=atom_name_2, coord_ref=coord_atom2,
                                                                    rcut=rcut_HostAtom_Wannier, step=step, Atom_ID=Atom_ID,
                                                                    print_degeneracy=print_degeneracy)
                                            
                                            degree_angle = angle(coord_wannier_near_cation, coord_atom2, coord_atom1)
                                                            
                                            # This will help in identifying a chemical bond
                                            if degree_angle >= AnlgeCut_HostWannier_Host_SecondaryAtom:
                                                count_number_of_secondary_atoms += 1
                                                Atom_ID_secondary_list.append(Atom_ID_secondary)
                                                Atom_ID_secondary_coordinates.append(coord_atom2)
                                                

                    #if count_number_of_wanniers_near_cation < 15:
                    BondAsymmetryValue = BondAsymmetry(count_number_of_secondary_atoms, coord_atom1, Atom_ID_secondary_coordinates)
                    if print_output:
                        if chargeAnalysis:
                            print(f'Step: {step:3d} \t Atom-ID: {Atom_ID:3d} \t Coordination: {count_number_of_secondary_atoms} \t' +
                                  f'Number-of-wanniers: {count_number_of_wannier_near_host_atom:2d} \t Charge: {round(charge[Atom_ID-1],4):.4f} \t' + 
                                  f'Bonding-Atom-ID: {Atom_ID_secondary_list} \t Asymmtery: {BondAsymmetryValue} \t XYZ: {coord_atom1}')
                        else: 
                            print(f'Step: {step:3d} \t Atom-ID: {Atom_ID:3d} \t Coordination: {count_number_of_secondary_atoms} \t' +
                                  f'Number-of-wanniers: {count_number_of_wannier_near_host_atom:2d} \t Bonding-Atom-ID: {Atom_ID_secondary_list} \t'+
                                  f'Asymmtery: {BondAsymmetryValue} \t XYZ: {coord_atom1}')
                        
                    if write_output:
                        fw = open(Directory +atom_name_1 +'-Structure-analysis-result.dat','a')
                        if chargeAnalysis:
                            fw.write(f'Step: {step} \t Atom-ID: {Atom_ID:3d} \t Coordination: {count_number_of_secondary_atoms} \t'+
                                     f'Number-of-wanniers: {count_number_of_wannier_near_host_atom:2d} \t Charge: {round(charge[Atom_ID-1],4):.4f}, \t' +
                                     f'Bonding-Atom-ID: {Atom_ID_secondary_list} \t Asymmtery: {BondAsymmetryValue} \t XYZ: {coord_atom1} \n')
                        else:
                            fw.write(f'Step: {step} \t Atom-ID: {Atom_ID:3d} \t Coordination: {count_number_of_secondary_atoms} \t' +
                                     f'Number-of-wanniers: {count_number_of_wannier_near_host_atom:2d} \t Bonding-Atom-ID: {Atom_ID_secondary_list}, \t' +
                                     f'Asymmtery: {BondAsymmetryValue} \t XYZ: {coord_atom1} \n')
                                 
                    Host_atom_coordination[count_number_of_secondary_atoms] += 1

        Wannier_Distance_Energy = np.array(Wannier_Distance_Energy, dtype='float')
        if plot_wannier_dist:
            plt.figure(figsize=(12,10))
            plt.hist(Wannier_Distance_Energy[:,0], bins=50)
            plt.xlabel('Distance', fontsize=15)
            plt.ylabel('count', fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.xlim(0,1)
            plt.title('Histogram of distances of Wannier centers to the host atom', fontsize=18)
            
        # Printing the coordiantion number of nfold 
        Total_Percentage_N_folds = 0
        Coordination_host_atom = 0
        print()
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('----------------------------------------')
        print('N_fold, count of N_fold, Percentage')
        print('----------------------------------------')
        for n_fold, value in enumerate(Host_atom_coordination):
            if value != 0 :
                percentage = value/(count_number_of_host_atoms * self.n_steps) * 100
                print(f'{n_fold}         {value/self.n_steps:>8}    {round(percentage,2):>8}')
                Total_Percentage_N_folds += percentage
                Coordination_host_atom += n_fold * value 
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print()
        print(f'Total {atom_name_1} counted is {Total_Percentage_N_folds}% and total average coordination number is {Coordination_host_atom/(count_number_of_host_atoms * self.n_steps)}.')
                                
        if write_output:
            # Writing data to a file
            # fw = open(Directory + 'Structure-analysis-result.dat','a')
            # Printing the coordiantion number of nfold 
            Total_Percentage_N_folds = 0
            Coordination_host_atom = 0
            fw.write('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
            fw.write('N_fold, count of N_fold, Percentage\n')
            fw.write('----------------------------------------\n')
            for n_fold, value in enumerate(Host_atom_coordination):
                if value != 0 :
                    percentage = value/(count_number_of_host_atoms * self.n_steps) * 100
                    fw.write(f'{n_fold}         {value/self.n_steps:>8}     {round(percentage,2):>8}\n')
                    Total_Percentage_N_folds += percentage
                    Coordination_host_atom += n_fold * value 
            fw.write('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n \n')
            fw.write(f'Total {atom_name_1} counted is {Total_Percentage_N_folds}% and total average coordination number is {Coordination_host_atom/(count_number_of_host_atoms * self.n_steps)}. \n')
            fw.close()

        return Wannier_Distance_Energy


    def get_farthest_atom(self, atom_name_search, coord_ref, rcut, step, Atom_ID, print_degeneracy=False):
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
        # Counting number of wannier centers near host atom in cut-off radius
        count_number_of_wanniers_near_host = 0

        # Storing coordinates of all the wannier centers near host atom 
        coord_of_various_wannier_centers_near_host_atom = []
        dist_of_various_wannier_centers_near_host_atom =  []

        # Atom_ID_secondary = 0 # Secondary Atom ID
        Atom_ID_Wannier = 0
        Atom_ID_Wannier_list = []
        Wannier_Distance_Energy = []
        for j, atom_wannier in enumerate(self.coordinates[step]):

            Atom_ID_Wannier += 1

            if atom_wannier[0] == atom_name_search:
                coord_atom_wannier_near_host = np.array(atom_wannier[1:], dtype=np.float_)
                _, dist_atom_wannier_near_host = displacement(coord_ref, coord_atom_wannier_near_host)

                # I assume no oxygen-wanniers are with in radium 1 Ang of host atom
                if dist_atom_wannier_near_host < rcut :   # Cutoff for Te-W 
                    count_number_of_wanniers_near_host += 1
                    coord_of_various_wannier_centers_near_host_atom.append(coord_atom_wannier_near_host)
                    dist_of_various_wannier_centers_near_host_atom.append(dist_atom_wannier_near_host)

                    Wannier_Distance_Energy.append(\
                    [dist_atom_wannier_near_host, self.coordinates_energy[step][Atom_ID_Wannier-1], Atom_ID])


        if count_number_of_wanniers_near_host > 1 :
            # print('WARNING: There are more that one wanniers centers near host atom. \
            # Thus selecting the one farthest.', end='\r')
            
            if print_degeneracy:
                if atom_name_1 == 'Tl' and count_number_of_wanniers_near_host < 6:
                    for i, value in enumerate(Atom_ID_Wannier_list):
                        print(f'WARNING: DEGENERACY {count_number_of_wanniers_near_host} -- Step: {step} \t ' +
                                f'Atom-ID: {Atom_ID} \t Atom-ID-Wannier: {value} \t ' +
                                f'Distance-Atom-Wannier: {round(dist_of_various_wannier_centers_near_host_atom[i], 6)} \t ' +
                                f'Energy: {self.coordinates_energy[step][value-1][0]}')
                elif atom_name_1 == 'Te':
                    for i, value in enumerate(Atom_ID_Wannier_list):
                        print(f'WARNING: DEGENERACY {count_number_of_wanniers_near_host} -- Step: {step} \t ' +
                                f'Atom-ID: {Atom_ID} \t Atom-ID-Wannier: {value} \t ' +
                                f'Distance-Atom-Wannier: {round(dist_of_various_wannier_centers_near_host_atom[i], 6)} \t ' +
                                f'Energy: {self.coordinates_energy[step][value-1][0]}')
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
            print(f'WARNING: There are no wanniers center near host atom with Atom ID {Atom_ID} in step {step}.')

        return dist_atom_wannier_near_host, coord_atom_wannier_near_host, count_number_of_wanniers_near_host, Wannier_Distance_Energy
