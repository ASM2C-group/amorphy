import os, sys, ast
import numpy as np
sys.path.append("/home/raghvp01/MD-traj-analysis-code/StructureAnalysis/")
import time, logo
from basis_reduction import minkowski_reduce
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
#Lattice = np.array([[16.6000003815,         0.0000000000,         0.0000000000],
#                    [ 0.0000000000,        11.0780000687,         0.0000000000],
#                    [ 0.0000000000,         0.0000000000,        15.7139997482]])

# Tl2Te3O7
#Lattice=np.array([[ 13.6780004501,         0.0000000000,         0.0000000000],
#                  [ -5.7719869934,        13.6975425414,         0.0000000000],
#                  [ -3.2214491920,        -1.7331681240,         9.2208890499]])

# Tl2Te2O5
#Lattice = np.array([[14.2379999161,         0.0000000000,         0.0000000000],
#                    [ 0.0000000000,        12.1379995346,         0.0000000000],
#                    [-6.9401691375,         0.0000000000,        15.3850884008]])


#print(A, B, C, ALPHA, BETA, GAMMA)
#sys.exit()


#A, B, C =  16.23892446618999,  16.23892446618999,  16.2389244661899 # x = 0.0
#A, B, C = 19.5662, 19.5662, 19.5662   # x =0.1
#A, B, C = 19.47335141773691, 19.47335141773691, 19.47335141773691 # x =0.2
#A, B, C = 19.38359280464976, 19.38359280464976, 19.38359280464976 # x = 0.3
#A, B, C = 20.069383346824402, 20.069383346824402, 20.069383346824402 # x =0.4
A, B, C = 19.9961464967, 19.9961464967, 19.9961464967 # x = 0.5
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
fileCharge = os.path.join(Directory + 'DDEC6_even_tempered_net_atomic_charges.xyz')
fileTraj = os.path.join(Directory + 'SiO2-pos-1.xyz')
##############################################

atom_name_1 = 'Te'  # Host atom symbol
atom_name_2 = 'Tl'   # Wannier center symbol
atom_name_3 = 'O'   # Secondary atatom_name_1

##############################################
##############################################
SKIP = 1
RESOLUTION = 500
##############################################

#--------------------------------------------#
#    Describing atomic cutoffs and name      #
#--------------------------------------------#
rcut_Tl_O = 2.8
rcut_Te_O = 2.5
rcut_HostAtom_Wannier = 1.0
rcut_HostAtom_SecondaryAtom = 2.5
rcut_tolerance_distance_selection = 0.05
AnlgeCut_HostWannier_Host_SecondaryAtom = 73  # degrees

#--------------------------------------------#
#   For Bond anlge distribution function     #
#--------------------------------------------#
bond_len_cut_pair12 = 1.0
bond_len_cut_pair23 = 2.8

##############################################


if __name__ == "__main__":

    #################################################################################
    sys.stderr.write(logo.art)                                                       
    if is_orthorhombic:                                                              
        sys.stderr.write('Cell is orthorhmobic, so uncomment njit decorator in '\
              'periodic_boundary_condition.py module to gain speed. \n \n')          
    else:                                                                            
        sys.stderr.write('If there is warning from numba, thus you need to comment '\
              'njit decorator in periodic_boundary_condition.py module. \n \n')      
    sys.stdout.flush()                                                               
    #################################################################################

    def rdf(write=False):
        from radial_and_bond_angle_distribution import PDF
        TeO2 = PDF()
        # Compute radial distribution function
        tik = time.perf_counter()
        TeO2.compute_volume_per_atom()
        TeO2.compute_radial_distribution(r_cutoff=0.5*min(A,B,C))
        TeO2.bond_plot(write_data=write)
        tok = time.perf_counter()
        print(f'\nTime elapsed : {tok-tik} seconds  ({(tok-tik)/60}) minutes.')

    def bdf(write=False):
        from radial_and_bond_angle_distribution import PDF
        # Compute bond analysis
        tik = time.perf_counter()
        TeO2 = PDF()
        TeO2.compute_volume_per_atom()
        TeO2.compute_bond_angle_distribution()
        TeO2.bond_angle_plot(write_data=write)
        tok = time.perf_counter()
        print(f'\nTime elapsed : {tok-tik} seconds  ({(tok-tik)/60}) minutes.')


    def writetraj(cutoff):
        from write_trajectory import WriteTrajectory

        tik = time.perf_counter()
        Config = WriteTrajectory()
        Config.clean_degenerate_wannier_atoms(rcut=cutoff)
        tok = time.perf_counter()
        print(f'\nTime elapsed : {tok-tik} seconds  ({(tok-tik)/60}) minutes.')
    

    def get_vmd_bonds(atom_ref, atom_search, rcut):
        from topology import vmd_bond_plugin
        
        tik = time.perf_counter()
        vmd_bond_plugin(atom_ref, atom_search, rcut)
        tok = time.perf_counter()
        print(f'\nTime elapsed : {tok-tik} seconds  ({(tok-tik)/60}) minutes.')

    def compute_all_distances(atom_name1, atom_name2, minmax):
        from topology import compute_all_distances
        tik = time.perf_counter()

        atom1_description = atom_name1.split('@')
        atom2_description = atom_name2.split('@')
        if len(atom1_description) == 1 and len(atom2_description) == 1:
            atom1_description = atom1_description[0]
            atom2_description = atom2_description[0]
            compute_all_distances(atom_name_1=atom1_description, 
            atom_name_2=atom2_description, minmax_stats=minmax)

        elif len(atom1_description) == 1 and len(atom2_description) == 2:
            atom2_description[1] = ast.literal_eval(atom2_description[1])
            assert isinstance(atom2_description[1], list)
            for index in atom2_description[1]:
                compute_all_distances(atom_name_1=atom1_description[0], 
                atom_name_2=atom2_description[0]+f'@{index}', minmax_stats=minmax)

        elif len(atom1_description) == 2 and len(atom2_description) == 1:
            atom1_description[1] = ast.literal_eval(atom1_description[1])
            assert isinstance(atom1_description[1], list)
            for index in atom1_description[1]:
                compute_all_distances(atom_name_1=atom1_description[0]+f'@{index}', 
                atom_name_2=atom2_description[0], minmax_stats=minmax)


        elif len(atom1_description) == 2 and len(atom2_description) == 2:
            atom1_description[1] = ast.literal_eval(atom1_description[1])
            atom2_description[1] = ast.literal_eval(atom2_description[1])
            assert isinstance(atom1_description[1], list)
            assert isinstance(atom2_description[1], list)
            for index in atom1_description[1]:
                for index2 in atom2_description[1]:
                    compute_all_distances(atom_name_1=atom1_description[0]+f'@{index}', 
                    atom_name_2=atom2_description[0]+f'@{index2}', minmax_stats=minmax)
        
        tok = time.perf_counter()
        print(f'\nTime elapsed : {tok-tik} seconds  ({(tok-tik)/60}) minutes.')

    def compute_coordination_number(atom1, atom2, rcut):
        from topology import compute_coordination

        tik = time.perf_counter()
        coordination = compute_coordination(atom_1=atom1, atom_2=atom2, rcut=rcut)
        for i in coordination:
            print('Constrained Frames  :', i)
        tok = time.perf_counter()
        print(f'\nTime elapsed : {tok-tik} seconds  ({(tok-tik)/60}) minutes.')

    def get_frameID_with_constraint(rcut_12, rcut_23, rcut_13, rcut_11, rcut_22, rcut_33):
        from topology import get_constrained_frameID

        tik = time.perf_counter()
        frames = get_constrained_frameID(atom_name_1, atom_name_2, atom_name_3, rcut_12, rcut_23, rcut_13, rcut_11, rcut_22, rcut_33)
        for i in frames:
            print('Constrained FrameID: ', i)
        tok = time.perf_counter()
        print(f'\nTime elapsed : {tok-tik} seconds  ({(tok-tik)/60}) minutes.')

    def wannier_cation_host(tlrcut,
                            write_dist_wannier=False, 
                            plot_histogram2D=False, 
                            plot_histogram_method='matplotlib', 
                            rcutoff_coordination=False,
                            plot_wannier_cation_anion_angle=False):

        from wannier_structural_analysis import WannierAnalysis
        if atom_name_1 == 'O' :
            raise NameError ('WARNING: Host atom is Oxygen')

        angle = [] 
        max_dist = []
        tik = time.perf_counter()
        TeO2 = WannierAnalysis()
        average_coordination, max_dist, angle, wannier_dist_ener = \
        TeO2.compute_neighbour_wannier(tlrcut=tlrcut,
                                       compute_qnm_statistics=True, 
                                       chargeAnalysis=False, method='DDEC6', 
                                       write_output=False, 
                                       print_output=False, 
                                       plot_wannier_dist=False,
                                       print_degeneracy=True)

        if plot_wannier_cation_anion_angle:
            plt.figure(figsize=[25,15])
            bin_heights, bin_borders, _  = plt.hist(angle, bins=180)
            bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
            plt.plot(bin_centers, bin_heights)
            plt.xticks(np.linspace(0,180,18))
            plt.savefig('x0.5-angle.png')
            plt.show()

        if write_dist_wannier:
            '''Very Unstable plot. Can't be trusted 😒
            '''
            x = wannier_dist_ener[:,0]
            y = wannier_dist_ener[:,1]

            with open('wannier-dist-ener-'+atom_name_1+'.dat', 'w') as fileOpen:
                fileOpen.write(f'# dist  energy \n')
                for i in range(len(x)):
                    fileOpen.write(f'{x[i]}   {y[i]} \n')

            if plot_histogram2D:
                if plot_histogram_method == 'snsplot':
                    from histogram2Dplot import snsplot
                    snsplot(x, y)
                elif plot_histogram_method == 'matplotlib':
                    from histogram2Dplot import matplotlib
                    matplotlib(x, y)

        # To create data file with coordination and rcut values
        # This is suppose to use with bash script rcut_coordination.sh 
        if rcutoff_coordination:
            rcut_off = rcut_HostAtom_SecondaryAtom
            with open('rcut-coordination.dat', 'a') as fileOpen:
                fileOpen.write(f'  {rcut_off}   {average_coordination}  \n ')

        tok = time.perf_counter()
        print(f'\nTime elapsed : {tok-tik} seconds ({(tok-tik)/60}) minutes.')

    def wannier_anion_host(write_dist_wannier=False, \
                           plot_histogram2D=False, plot_histogram_method='matplotlib'):
        from wannier_structural_analysis import WannierAnalysis

        # Computing Wannier with host atom as anion
        #if atom_name_1 != 'O' :
        #    raise NameError ('WARNING: Host atom is not Oxygen')

        tik = time.perf_counter()
        TeO2 = WannierAnalysis()
        wannier_dist_ener = \
        TeO2.compute_neighbour_wannier_host_anion(chargeAnalysis=False, method='DDEC6',\
                                                write_output=True, print_output=False, \
                                                plot_wannier_dist=False)
        
        if write_dist_wannier:
            '''Very Unstable plot. Can't be trusted 😒
            ''' 
            x = wannier_dist_ener[:,0]
            y = wannier_dist_ener[:,1]

            with open('wannier-dist-ener-'+atom_name_1+'.dat', 'w') as fileOpen:
                fileOpen.write(f'# dist  energy  \n')
                for i in range(len(x)):
                    fileOpen.write(f'{x[i]}   {y[i]} \n')

            if plot_histogram2D:
                if plot_histogram_method == 'snsplot':
                    from histogram2Dplot import snsplot
                    snsplot(x, y)
                elif plot_histogram_method == 'matplotlib':
                    from histogram2Dplot import matplotlib
                    matplotlib(x, y)

        tok = time.perf_counter()
        print(f'\nTime elapsed : {tok-tik} seconds  ({(tok-tik)/60}) minutes.')
     

    # non_bonding_oxygen = non_bonding_oxygen.split()
    # compute_all_distances(atom_name1=f'Te', atom_name2=f'Tl', minmax=True)
    # compute_coordination_number(atom1='Tl', atom2=non_bonding_oxygen, rcut=3.2)
    get_frameID_with_constraint(rcut_12=1.0,rcut_23=2.2, rcut_13=1.0, rcut_11=3.0, rcut_22=1.0, rcut_33=2.2)
    # get_vmd_bonds('Te', 'O', rcut=2.5)
    # rdf(write=True)
    # bdf(write=True)
    # writetraj(cutoff=1.2)
    #wannier_cation_host(write_dist_wannier=False, rcutoff_coordination=False, plot_wannier_cation_anion_angle=False, plot_histogram2D=False, plot_histogram_method='snsplot',  tlrcut=3.2)
    # wannier_anion_host(write_dist_wannier=False, plot_histogram2D=False, plot_histogram_method='snsplot')
