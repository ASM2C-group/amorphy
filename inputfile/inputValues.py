import os, sys, ast
import numpy as np
sys.path.append("/home/raghvp01/MD-traj-analysis-code/Structure_Analysis")
import time, logo, tqdm
from elemental_data import atomic_no, atomic_symbol, get_atomic_IDs
from basis_reduction import minkowski_reduce
from decorators import timeit
from numba.typed import List
from numba.core.errors import NumbaWarning, NumbaExperimentalFeatureWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
from geometric_properties import (
    cellMatrix_to_cellParameter, cellParameter_to_cellMatrix, 
    check_orthorhombic)
import matplotlib.pyplot as plt

''' 
Created with packages 
__numpy.version__ = '1.21.0'
__matplotlib.version__ = '3.2.0'
__numba.version__ = '0.55.1'
'''

######################################################################################

# Tl2TeO3
#LatticeMatrix = np.array([[16.6000003815,         0.0000000000,         0.0000000000],
#                          [ 0.0000000000,        11.0780000687,         0.0000000000],
#                          [ 0.0000000000,         0.0000000000,        15.7139997482]])

# Tl2Te3O7
#LatticeMatrix=np.array([[ 13.6780004501,         0.0000000000,         0.0000000000],
#                        [ -5.7719869934,        13.6975425414,         0.0000000000],
#                        [ -3.2214491920,        -1.7331681240,         9.2208890499]])

# Tl2Te2O5
#LatticeMatrix = np.array([[14.2379999161,         0.0000000000,         0.0000000000],
#                          [ 0.0000000000,        12.1379995346,         0.0000000000],
#                          [-6.9401691375,         0.0000000000,        15.3850884008]])


# Binary system
#A, B, C = 16.23892446618999,   16.23892446618999,  16.2389244661899    # x = 0.0
#A, B, C = 19.5662,             19.5662,            19.5662             # x = 0.1
#A, B, C = 19.47335141773691,   19.47335141773691,  19.47335141773691   # x = 0.2
#A, B, C = 19.38359280464976,   19.38359280464976,  19.38359280464976   # x = 0.3
#A, B, C = 20.069383346824402,  20.069383346824402, 20.069383346824402  # x = 0.4
#A, B, C = 19.9961464967,       19.9961464967,      19.9961464967       # x = 0.5


# Ternary system
A, B, C = 21.0258283859536,    21.0258283859536,    21.0258283859536        # x=0.05, y=0.2
#A, B, C = 20.933504435425377,  20.933504435425377,  20.933504435425377      # x=0.05, y=0.3
#A, B, C = 20.816067482029105,  20.816067482029105,  20.816067482029105      # x=0.05, y=0.4
#A, B, C = 19.52325657542347,   19.52325657542347,   19.52325657542347       # x=0.1 , y=0.1
#A, B, C = 20.872736621246155,  20.872736621246155,  20.872736621246155      # x=0.1 , y=0.2
#A, B, C = 20.778905383335992,  20.778905383335992,  20.778905383335992      # x=0.1 , y=0.3

# Comment these two lines for LatticeMatrix
ALPHA, BETA, GAMMA = 90.0, 90.0, 90.0
LatticeMatrix = cellParameter_to_cellMatrix([A,B,C,ALPHA,BETA,GAMMA])

A, B, C, ALPHA, BETA, GAMMA = cellMatrix_to_cellParameter(LatticeMatrix)
is_orthorhombic = check_orthorhombic(LatticeMatrix)
minkowski_reduce_cell = minkowski_reduce(LatticeMatrix)
##############################################

#--------------------------------------------#
#          Directory of the files            #
#--------------------------------------------#
Directory = os.getcwd()
if Directory[-1] != '/': Directory = Directory + '/'
fileCharge = os.path.join(Directory + '../DDEC6_even_tempered_net_atomic_charges.xyz')
fileTraj = os.path.join(Directory + '../TeO2-HOMO_centers.xyz')
##############################################

atom_name_1 = 'Ti'  # Host atom symbol
atom_name_2 = 'X'   # Wannier center symbol
atom_name_3 = 'O'   # Secondary atom_name

##############################################
##############################################
SKIP = 1
RESOLUTION = 500
##############################################

#--------------------------------------------#
#    Describing atomic cutoffs parameters    #
#--------------------------------------------#
rcut_HostAtom_Wannier = 0.8
rcut_HostAtom_SecondaryAtom = 2.5
rcut_tolerance_distance_selection = np.inf
AnlgeCut_HostWannier_Host_SecondaryAtom = 0.0

#--------------------------------------------#
#   For Bond anlge distribution function     #
#--------------------------------------------#
bond_len_cut_pair12 = 1.0
bond_len_cut_pair23 = 2.8

##############################################

#####################################################################################
##                                                                                 ## 
##                             Computating Functions                               ##
##                                                                                 ##
#####################################################################################
if __name__ == "__main__":

    #################################################################################
    sys.stderr.write(logo.art)                                                       
    sys.stdout.flush()                                                               
    from  system_description import SystemInfo
    SystemInfo()
    #################################################################################

    @timeit
    def get_rdf(Histogram, binwidth=0.005, write=False):
        from radial_and_bond_angle_distribution import PDF
        TeO2 = PDF()
        # Compute radial distribution function
        if Histogram == False:
            TeO2.compute_volume_per_atom()
            TeO2.compute_radial_distribution(r_cutoff=0.5*min(A,B,C))
            TeO2.bond_plot(write_data=write)
        else:
            TeO2.radial_distribution_histogram(atom_name_1, atom_name_2, normalized=True, binwidth=binwidth)

    @timeit
    def get_bdf(write=False):
        from radial_and_bond_angle_distribution import PDF
        # Compute bond analysis
        TeO2 = PDF()
        TeO2.compute_volume_per_atom()
        TeO2.compute_bond_angle_distribution()
        TeO2.bond_angle_plot(write_data=write)


    @timeit
    def writetraj(sequence_traj):
        from write_trajectory import WriteTrajectory
        Traj = WriteTrajectory()
        if sequence_traj:
            sequencing = Traj.sequencing_trajectory()
    
    @timeit
    def get_all_distances(atom_name1, atom_name2, minmax, step=0):
        from topology import compute_all_distances
        from read_trajectory import Trajectory

        atom_Data = Trajectory(filename=fileTraj)
        coordinates = atom_Data.coordinates[step]

        compute_all_distances(coordinates, atom1 = atom_name1, atom2=atom_name2, minmax_stats=minmax) 


    @timeit
    def get_coordination_number(coordinates, atom1, atom2, rcut, step=0):
        from topology import compute_coordination
        #from read_trajectory import Trajectory

        #atom_Data = Trajectory(filename=fileTraj)
        #coordinates = atom_Data.coordinates[step]

        if isinstance(atom2, list):
            numba_list_atom2 = List()
            [numba_list_atom2.append(x) for x in atom2]
        else:
            numba_list_atom2 = atom2

        l_fold, n_coordination = compute_coordination(coordinates=coordinates, atom_1=atomic_no(atom1), atom_2=numba_list_atom2, rcut=rcut)

        #print(n_coordination)
        #for i in coordination:
        #    print('Constrained Frames  :', i)
        #print("l_fold in percentage: ", l_fold, " coordination: ",  n_coordination)
        return l_fold, n_coordination

    @timeit
    def get_frameID_with_constraint(rcut_12, rcut_23, rcut_13, rcut_11, rcut_22, rcut_33):
        from topology import get_constrained_frameID

        frames = get_constrained_frameID(atom_name_1, atom_name_2, atom_name_3, rcut_12, rcut_23, rcut_13, rcut_11, rcut_22, rcut_33)
        for i in frames:
            print('Constrained FrameID: ', i)

    @timeit
    def wannier_cation_host(rcutoff_coordination=False):
        from wannier_structural_analysis import WannierAnalysis
        if atom_name_1 == 'O' :
            raise NameError ('WARNING: Host atom is Oxygen')

        TeO2 = WannierAnalysis()
        average_coordination = TeO2.compute_neighbour_wannier_host_cation(
                                       compute_qnm_statistics=False, print_BO_NBO=False, 
                                       chargeAnalysis=False, method='DDEC6', 
                                       write_output=True, print_output=False, 
                                       print_degeneracy=True)

        # To create data file with coordination and rcut values
        # This is suppose to use with bash script rcut_coordination.sh 
        if rcutoff_coordination:
            rcut_off = rcut_HostAtom_SecondaryAtom
            #rcut_off = AnlgeCut_HostWannier_Host_SecondaryAtom
            with open('rcut-coordination.dat', 'a') as fileOpen:
                fileOpen.write(f'  {rcut_off}   {average_coordination}  \n ')


    @timeit
    def wannier_anion_host():
        from wannier_structural_analysis import WannierAnalysis

        # Computing Wannier with host atom as anion
        if atom_name_1 != 'O' :
            raise NameError ('WARNING: Host atom is not Oxygen')

        TeO2 = WannierAnalysis()
        TeO2.compute_neighbour_wannier_host_anion(chargeAnalysis=True, method='DDEC6',
                                                  write_output=False, print_output=False)

    @timeit
    def ground_center_molecule(ID0, ID1, ID2, step=0):
        '''ID1 is the center atom to be ground first.
        '''
        from topology import ground_the_molecule
        from read_trajectory import Trajectory
        atom_Data = Trajectory(filename=fileTraj)
        coordinates = atom_Data.coordinates[step]
        
        coord = ground_the_molecule(coordinates=coordinates, ID0=ID0, ID1=ID1, ID2=ID2)
        print(len(coord))
        print("Lattice: ",A, B, C)
        for i in coord:
            print(f'{atomic_symbol(i[0]):>2}  {round(i[1], 6):>8}  {round(i[2], 6):>8}  {round(i[3], 6):>8}')
    
    @timeit
    def get_wrapped_atom(step=0):

        from topology import wrap_atoms
        from read_trajectory import Trajectory
        atom_Data = Trajectory(filename=fileTraj)
        coordinates = atom_Data.coordinates[step]

        coord = wrap_atoms(coordinates=coordinates, cell=LatticeMatrix)
        print(len(coord))
        print("Lattice: ",A, B, C)
        for i in coord:
            print(f'{atomic_symbol(i[0]):>2}  {round(i[1], 6):>8}  {round(i[2], 6):>8}  {round(i[3], 6):>8}')

    @timeit
    def hydrogen_passivation(step=0):
       
        from topology import hydrogen_passivate
        from read_trajectory import Trajectory
        atom_Data = Trajectory(filename=fileTraj)
        coordinates = atom_Data.coordinates[step]
        
        coordinate, hydrogen_coordinate = hydrogen_passivate(coordinates=coordinates, cutoff=2.4)
        coordinates = np.concatenate((coordinate, hydrogen_coordinate), axis=0)

        print(len(coordinates))
        print("Lattice: ", A, B, C)
        for i in coordinates:
             print(f'{atomic_symbol(i[0]):>2}  {round(float(i[1]), 6):>8}  {round(float(i[2]), 6):>8}  {round(float(i[3]), 6):>8}')

    @timeit
    def get_neighbor_list(atom, rcut=3.26):
        from topology import neighbor_list
        from read_trajectory import Trajectory
        atom_Data = Trajectory(filename=fileTraj)
        step=0
        coordinates = atom_Data.coordinates[step]
        print(neighbor_list(coordinates, atomic_species=atom, rcut=rcut))

    @timeit
    def get_BO_NBO_coordination(atom_name_1, cut, charge_analysis=False):
        from topology import compute_BO_NBO_coordination
        from collections import defaultdict
        from read_trajectory import Trajectory
        from charge_analysis import ChargeAnalysis
            
        atom_Data = Trajectory(filename=fileTraj)
        
        if charge_analysis:
            TeO2 = ChargeAnalysis()
            with open(fileCharge, 'r') as openFileCharge:
                 TeO2.dataCharge = openFileCharge.readlines()
        
        try:
            BO_step, BO_ID = np.loadtxt('BO.dat', usecols=(1,5), unpack=True, dtype=int)
            NBO_step, NBO_ID = np.loadtxt('NBO.dat', usecols=(1,5), unpack=True, dtype=int)
        except FileNotFoundError:
            print('BO.dat and NBO.dat does not exist. Please first run the wannier_cation_host with print_BO_NBO=True')

        
        BO_info = defaultdict(list)
        for i, j in zip(BO_step, BO_ID):
            BO_info[i].append(j)


        NBO_info = defaultdict(list)
        for i, j in zip(NBO_step, NBO_ID):
            NBO_info[i].append(j)
        
        
        with open('Tl-BO-NBO-coordination.dat', 'w') as fw:
            coord_O, coord_BO, coord_NBO = [], [], []
            l_count, l_fold , l_fold_BO_NBO = [], [], []
            total_lfold_BO ,total_lfold_NBO = [], []

            oxygen_IDs = get_atomic_IDs(atom_Data.coordinates[0],'O')
            
            for steps, ids  in BO_info.items():
                coordinates = atom_Data.coordinates[steps]
                nbo_ids = [element for element in oxygen_IDs if element not in ids ]
                
                if charge_analysis:
                    charges = TeO2.chargeAnalysis(fileCharge=TeO2.dataCharge, step=steps, method='DDEC6')
                    coordination_O, lcount, lfold, l_BO_NBO = compute_BO_NBO_coordination(coordinates, atom1=atom_name_1, BO_ID=list(ids), NBO_ID=list(nbo_ids), rcut=cut, charge=charges)
                else:
                    coordination_O, lcount, lfold, l_BO_NBO = compute_BO_NBO_coordination(coordinates, atom1=atom_name_1, BO_ID=list(ids), NBO_ID=list(nbo_ids), rcut=cut)

                coord_O.append(coordination_O)
                l_count.append(lcount)
                l_fold.append(lfold)
                l_fold_BO_NBO.append(l_BO_NBO)

                l_fold_BO, coordination_BO = get_coordination_number(coordinates, atom1=atom_name_1, atom2=ids, rcut=cut)
                coord_BO.append(coordination_BO)
                total_lfold_BO.append(l_fold_BO)
                
                l_fold_NBO, coordination_NBO = get_coordination_number(coordinates, atom1=atom_name_1, atom2=nbo_ids, rcut=cut)
                coord_NBO.append(coordination_NBO)
                total_lfold_NBO.append(l_fold_NBO)

            n_coord_O , n_coord_BO, n_coord_NBO  = np.mean(coord_O),  np.mean(coord_BO),  np.mean(coord_NBO)
            n_coord_O_std , n_coord_BO_std, n_coord_NBO_std  = np.std(coord_O), np.std(coord_BO),  np.std(coord_NBO)
            
            total_lcount_O_mean = np.mean(l_count, axis=0)
            total_lcount_O_std = np.std(l_count, axis=0)

            total_lfold_O_mean = np.mean(l_fold, axis=0)
            total_lfold_O_std  = np.std(l_fold, axis=0)
            
            total_lfold_BO_mean  = np.mean(total_lfold_BO, axis=0)
            total_lfold_NBO_mean = np.mean(total_lfold_NBO, axis=0)

            total_lfold_BO_std  = np.std(total_lfold_BO, axis=0)
            total_lfold_NBO_std = np.std(total_lfold_NBO, axis=0)

            total_lfold_BO_NBO_mean = np.mean(l_fold_BO_NBO, axis=0)
            total_lfold_BO_NBO_std  = np.std(l_fold_BO_NBO, axis=0)
            
            fw.write(f'#    l-fold       count            Percentage (O)    p-l-fold (BO)        p-l-fold (NBO)      Percentage     \t | \t  Percentage (BO)  \t   Percentage (NBO) \n')

            for index, values in enumerate(zip(total_lcount_O_mean, total_lcount_O_std, total_lfold_O_mean, total_lfold_O_std, total_lfold_BO_mean, total_lfold_BO_std, total_lfold_NBO_mean, total_lfold_NBO_std)):
                if np.array(values).any() > 0:
                    fw.write(f'{index:>8}   {values[0]:>8.3f} ± {values[1]:<5.3f}    {values[2]:>8.3f} ± {values[3]:<5.3f} %  {"|":>63}    {values[4]:>5.3f} ± {values[5]:>5.3f}  \t    {values[6]:>5.3f} ± {values[7]:>5.3f} \n')
            
                    for i in range(len(total_lfold_BO_NBO_mean)):
                        for j in range(len(total_lfold_BO_NBO_mean)):
                            for k in range(len(total_lfold_BO_NBO_mean)):
                                if total_lfold_BO_NBO_mean[i,j,k] != 0:
                                     if i == index:
                                        fw.write(f'  {j:>57}    {k:>17}    {total_lfold_BO_NBO_mean[i, j, k]:>16.3f} ± {total_lfold_BO_NBO_std[i, j, k]:>5.3f} % \t | \n')

            fw.write(f'\nRcutoff = {cut:>4.2f} Å,    n_{atom_name_1}_O = {n_coord_O:>5.3f} ± {n_coord_O_std:<5.3f} {"|":>70}'+
                     f' \t n_{atom_name_1}_BO = {n_coord_BO:>5.3f} ± {n_coord_BO_std:<5.3f} ,  n_{atom_name_1}_NBO = {n_coord_NBO:>5.3f} ± {n_coord_NBO_std:<5.3f} \n')

    #######################################################################################################################
    # get_BO_NBO_coordination('Tl', cut=3.26, charge_analysis=False)
    # get_neighbor_list(atom='Tl')
    # hydrogen_passivation()
    # get_wrapped_atom()
    # ground_center_molecule(ID0=0, ID1=1, ID2=2)
    # get_frameID_with_constraint(rcut_12=1.0,rcut_23=2.2, rcut_13=1.0, rcut_11=3.0, rcut_22=1.0, rcut_33=2.2)
    # get_coordination_number(atom1='Tl', atom2=oxyg, rcut=2.8,  step=0)
    # get_rdf(Histogram=True, binwidth=0.005, write=True)
    # get_bdf(write=True)
    # writetraj(sequence_traj=True)
    wannier_cation_host(rcutoff_coordination=False)
    # wannier_anion_host()
