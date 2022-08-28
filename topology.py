import numpy as np
from read_trajectory import Trajectory
from inputValues import fileTraj, Directory
from geometric_properties import rotate
from distance_minkowski_reduction import wrap_positions
from periodic_boundary_condition import displacement, angle
from tqdm import  tqdm


def translate():
    pass


def wrap_atoms(cell):
    '''Wrap the atoms in the box.

       Parameters:
       --------------------------------------------------------
       cell : 3*3 lattice matrix

       Result:
       --------------------------------------------------------
       coordinates : wrapped atoms configuration

    '''
    atom_Data = Trajectory(filename=fileTraj)
    coordinates = atom_Data.coordinates[0]

    for atom_ID, value in enumerate(coordinates):
        coord = np.atleast_2d(value[1:])
        coordinates[atom_ID][1:] = wrap_positions(np.asarray(coord, dtype=np.float_), cell)[0]

    return coordinates

def hydrogen_passivate():
    pass

def ground_the_molecule(ID0, ID1, ID2):
    ''' Ground the molecule in xy-plane (convention) annd put it
        close to center in the xy plane.

        Parameters:
        --------------------------------------------------------
        ID0 = ID of atom 1
        ID1 = ID of atom 2 (Centered atom in xy-plane)
        ID2 = ID of atom 3
     
        Return:
        --------------------------------------------------------
        coord = coodinates of grounded molecule

    '''
    assert isinstance(ID0, int)
    assert isinstance(ID1, int)
    assert isinstance(ID2, int)

    atom_Data = Trajectory(filename=fileTraj)
    coordinates = atom_Data.coordinates

    # 1. Grabbing x,y,z coordinate of ID1, ID0 & ID2  atom
    coord_atom_0 = np.array(coordinates[0][ID0][1:], dtype=np.float_)                     
    coord_atom_1 = np.array(coordinates[0][ID1][1:], dtype=np.float_)
    coord_atom_2 = np.array(coordinates[0][ID2][1:], dtype=np.float_)
    
    coord = coordinates[0]

    # 2. Shifting the molecule with the ID0 atom placed on z=0 plane.
    for atom_ID, value in enumerate(coordinates[0]):
        coord[atom_ID][1:] = coord[atom_ID][1:]-[0,0,coord_atom_1[2]]
    
   
    coord_atom_0 = np.array(coord[ID0][1:], dtype=np.float_)                     
    coord_atom_1 = np.array(coord[ID1][1:], dtype=np.float_)
    coord_atom_2 = np.array(coord[ID2][1:], dtype=np.float_)

    # 3. Fixing atom ID1, and calculating angle between ID0, ID1 
    #    & projection of ID0 on z=0 plane.
    proj_0 = coord_atom_0 - [0, 0, coord_atom_0[2]]
    Angle01 = angle(coord_atom_0, coord_atom_1, proj_0)

    perp_to_proj_0 = [-(proj_0[1]  - coord_atom_1[1])/(proj_0[0] - coord_atom_1[0]), 1, 0]

    # 4. Now rotate the whole molecule with angle "Angle12" 
    #    to bring atom ID1 on z=0 plane.
    for atom_ID, value in enumerate(coord): 
        coord[atom_ID][1:] = rotate(angle=Angle01, around_vector=perp_to_proj_0, position=coord[atom_ID][1:], center=coord_atom_1, radian=False)

    coord_atom_0 = np.array(coord[ID0][1:], dtype=np.float_)                     
    coord_atom_1 = np.array(coord[ID1][1:], dtype=np.float_)
    coord_atom_2 = np.array(coord[ID2][1:], dtype=np.float_)
    
    # 5. Find the normal vector of plane passing through ID0, ID1, ID2.
    #    Compute angle between the normal vector and z=0 plane.
    vector10 = coord_atom_0 - coord_atom_1
    vector21 = coord_atom_2 - coord_atom_1

    normal_vector = np.cross(vector10, vector21)
    Angle_plane = np.arccos(np.clip(np.dot(normal_vector, [0,0,1])/np.linalg.norm(normal_vector), -1, 1))
    

    # 6. Rotate the plane with ID0, ID1, ID2 atom to coincide with plane z=0.
    for atom_ID, value in enumerate(coord):
        coord[atom_ID][1:] = rotate(angle=Angle_plane, around_vector=vector10, position=coord[atom_ID][1:], center=coord_atom_1, radian=True)

    return coord

def get_constrained_frameID(atom_name_1, atom_name_2, atom_name_3, 
                          rcut_12=float(0), 
                          rcut_23=float(0), 
                          rcut_13=float(0),
                          rcut_11=float(0),
                          rcut_22=float(0),
                          rcut_33=float(0)):
    
    '''Get the frame ID (step number) with all the constraint specified.

       Constraint : All frame ID which have cutoff length between atoms 
                    larger than the cutoff supplied.
    
       Parameters:
       --------------------------------------------------------------------------
       rcut_12 : cut off distace between atom_name_1 and atom_name_2  
       rcut_23 : cut off distace between atom_name_2 and atom_name_3 
       rcut_13 : cut off distace between atom_name_1 and atom_name_3

       Return:
       --------------------------------------------------------------------------
       constrained_frames : list of constrained frameID
    '''
    
    atom_Data = Trajectory(filename=fileTraj)
    coordinates = atom_Data.coordinates
    n_steps = atom_Data.n_steps

    constrained_frames = []
    
    for step in tqdm(range(n_steps)):
        
        MinDist_12 = np.inf
        MinDist_23 = np.inf
        MinDist_13 = np.inf
        MinDist_11 = np.inf
        MinDist_22 = np.inf
        MinDist_33 = np.inf

        for atom_ID, value in enumerate(coordinates[step]):
            coord_1 = np.array(value[1:], dtype=np.float_)
    
            for atom_ID2, value2 in enumerate(coordinates[step]):
                coord_2 = np.array(value2[1:], dtype=np.float_)
    
                _, distance = displacement(coord_1, coord_2)
   
                if distance > 0:
                    if   value[0] == atom_name_1 and value2[0] == atom_name_2 and distance <= MinDist_12:  MinDist_12 = distance ; continue 
                    elif value[0] == atom_name_2 and value2[0] == atom_name_3 and distance <= MinDist_23:  MinDist_23 = distance ; continue
                    elif value[0] == atom_name_1 and value2[0] == atom_name_3 and distance <= MinDist_13:  MinDist_13 = distance ; continue
                    elif value[0] == atom_name_1 and value2[0] == atom_name_1 and distance <= MinDist_11:  MinDist_11 = distance ; continue
                    elif value[0] == atom_name_2 and value2[0] == atom_name_2 and distance <= MinDist_22:  MinDist_22 = distance ; continue
                    elif value[0] == atom_name_3 and value2[0] == atom_name_3 and distance <= MinDist_33:  MinDist_33 = distance ; continue

        print(f'Step: {step} \t' + 
              f'Mininum distance ({atom_name_1}-{atom_name_2}: {round(MinDist_12, 6)} \t' +
              f'{atom_name_2}-{atom_name_3}: {round(MinDist_23, 6)} \t' +
              f'{atom_name_1}-{atom_name_3}: {round(MinDist_13, 6)} \t' +
              f'{atom_name_1}-{atom_name_1}: {round(MinDist_11, 6)} \t' +
              f'{atom_name_2}-{atom_name_2}: {round(MinDist_22, 6)} \t' +
              f'{atom_name_3}-{atom_name_3}: {round(MinDist_33, 6)})')

        if (MinDist_12 < rcut_12 or  
            MinDist_23 < rcut_23 or 
            MinDist_13 < rcut_13 or 
            MinDist_11 < rcut_11 or 
            MinDist_22 < rcut_22 or 
            MinDist_33 < rcut_33):
            continue
        else:
           constrained_frames.append(step)

    return constrained_frames

def compute_coordination(atom_1, atom_2, rcut, step=0):

    ''' Compute the coordination number of atom_1 with respect to atom_2 within 
        cut-off radius rcut.

        Parameters:
        -------------------------------------------------------------------------
        atom_1 : This is the host atom.
            |______ : type ==> str

        atom_2 : This is the secondary atom.
            |______ : type ==> list or str

        rcut   : Radius of the coordination sphere
            |______ : type ==> float

        step   : Index of the n'th frame.
            |______ : type ==> int
            |______ : default value ==> 0 (Initial configuration)

        Return:
        -------------------------------------------------------------------------
        n_coordination : Coordination number of atom_1
            |______ : type ==> 1 dimensional nd.array
    '''

    atom_Data = Trajectory(filename=fileTraj)
    
    assert isinstance(atom_1, str)
    
    coordinates = atom_Data.coordinates[step]

    l_fold = np.zeros(12) # I assume no atom has more than 12 bonds.

    count_of_atom1 = 0

    if isinstance(atom_2, list):
        atom_2 = np.array(atom_2, dtype=np.int_)
        for atom_ID, value in enumerate(coordinates):
            
             if value[0] == atom_1:
                coord_atom_1 = np.array(value[1:], dtype=np.float_)
                count_of_atom1 += 1

                coordination = 0
                for atom_ID_second, value2 in enumerate(coordinates):
                    
                    if atom_ID_second in atom_2:
                        coord_atom_2 = np.array(value2[1:], dtype=np.float_)

                        _, distance = displacement(coord_atom_1, coord_atom_2)

                        if distance < rcut:
                            coordination += 1

                l_fold[coordination] += 1 

    elif isinstance(atom_2, str):
        for atom_ID, value in enumerate(coordinates):
            
            if value[0] == atom_1:
                coord_atom_1 = np.array(value[1:], dtype=np.float_)
                count_of_atom1 += 1

                coordination = 0
                for atom_ID_second, value2 in enumerate(coordinates):
                    
                    if value2[0] == atom_2:
                        coord_atom_2 = np.array(value2[1:], dtype=np.float_)

                        _, distance = displacement(coord_atom_1, coord_atom_2)
                        
                        if distance < rcut:
                            coordination += 1
 
                l_fold[coordination] += 1 

    else:
        raise TypeError(f'Type of atom_2: ({atom_2}) is not compatible with the funciton.')

    
    n_coordination = 0
    for fold, value in enumerate(l_fold):
        n_coordination += fold * value  

    n_coordination = n_coordination / count_of_atom1

    return l_fold, n_coordination

def compute_all_distances(atom_name_1,
                          atom_name_2,
                          minmax_stats=False,
                          step_Constrained=False):

    ''' This function computes the distances between all the atom 1 and among all 
        the atoms of type 2.

        Parameters:
        -------------------------------------------------------------------------
        atom_name_1 : Symbol of Host atom 
                      If the symbol of the system is followed by @ symbol
                      then atom_ID is expected.
        atom_name_2 : Symbol of Host atom 
                      If the symbol of the system is followed by @ symbol
                      then atom_ID is expected.
        minmax_stats: Prints the Atom ID of host and secondary with minimum
                      distance and maximum distance.

        Return:
        -------------------------------------------------------------------------
        null
    '''

    atom_name_1 = atom_name_1.split('@')
    atom_name_2 = atom_name_2.split('@')

    fname = Directory+atom_name_1[0]+'-'+atom_name_2[0]+'-distances.dat'
    with open(fname, 'a') as fw:
        atom_Data = Trajectory(filename=fileTraj)
        #fw.write(f'\n # Host-ID Neighbour-ID {atom_name_1}-{atom_name_2}-distances')

        Host_Secondary_Distances = []

        for step in range(atom_Data.n_steps):

            #progressBar = "\rProgress: " + ProgressBar(atom_Data.n_steps -1 , step, 100)
            #ShowBar(progressBar)
            
            if isinstance(step_Constrained, int) and step_Constrained != step:
                continue

            if len(atom_name_1) == 2:
                 
                for atom_ID, value in enumerate(atom_Data.coordinates[step]):
                    if  value[0] == atom_name_1[0] and atom_ID == int(atom_name_1[1]):
                        coord_atom_1 = np.array(value[1:], dtype=np.float_)
    
                        for atom_ID_2, value2 in enumerate(atom_Data.coordinates[step]):
                            if len(atom_name_2) == 2:

                                if value2[0] == atom_name_2[0] and atom_ID_2 == int(atom_name_2[1]):                                     
                                    coord_atom_2 = np.array(value2[1:], dtype=np.float_)
                                    _, distance = displacement(coord_atom_1, coord_atom_2)
                                    Host_Secondary_Distances.append([atom_ID, atom_ID_2, distance])
                                    fw.write(f'\n {atom_name_1[0]}: {atom_ID:>3} \t {atom_name_2[0]}: {atom_ID_2:>3} \t Distance: {round(float(distance), 6)} ')

                            elif len(atom_name_2) == 1:
                                if value2[0] == atom_name_2[0]: 
                                    coord_atom_2 = np.array(value2[1:], dtype=np.float_)
                                    _, distance = displacement(coord_atom_1, coord_atom_2)
                                    Host_Secondary_Distances.append([atom_ID, atom_ID_2, distance])
                                    fw.write(f'\n {atom_name_1[0]}: {atom_ID:>3} \t {atom_name_2[0]}: {atom_ID_2:>3} \t Distance: {round(float(distance), 6)} ')

                Host_Secondary_Distances = np.array(Host_Secondary_Distances)

                if minmax_stats:
                    minimum_dist = Host_Secondary_Distances[np.argmin(Host_Secondary_Distances[:,2])]
                    maximum_dist = Host_Secondary_Distances[np.argmax(Host_Secondary_Distances[:,2])]
                    fw.write(f'\nMin-dist ({atom_name_1[0]} {int(minimum_dist[0])} - {atom_name_2[0]} {int(minimum_dist[1])}): {minimum_dist[2]} \t'+
                             f'Max-dist ({atom_name_1[0]} {int(maximum_dist[0])} - {atom_name_2[0]} {int(maximum_dist[1])}): {maximum_dist[2]}')

            elif len(atom_name_1) == 1:
                for atom_ID, value in enumerate(atom_Data.coordinates[step]):
                    if  value[0] == atom_name_1[0]:
                        coord_atom_1 = np.array(value[1:], dtype=np.float_)
    
                        for atom_ID_2, value2 in enumerate(atom_Data.coordinates[step]):
                            if len(atom_name_2) == 2:
                                if value2[0] == atom_name_2[0] and atom_ID_2 == int(atom_name_2[1]): 
                                    coord_atom_2 = np.array(value2[1:], dtype=np.float_)
                                    _, distance = displacement(coord_atom_1, coord_atom_2)
                                    Host_Secondary_Distances.append([atom_ID, atom_ID_2, distance])
                                    fw.write(f'\n {atom_name_1[0]}: {atom_ID:>3} \t {atom_name_2[0]}: {atom_ID_2:>3} \t Distance: {round(float(distance), 6)} ')

                            elif len(atom_name_2) == 1:
                                if value2[0] == atom_name_2[0]: 
                                    coord_atom_2 = np.array(value2[1:], dtype=np.float_)
                                    _, distance = displacement(coord_atom_1, coord_atom_2)
                                    Host_Secondary_Distances.append([atom_ID, atom_ID_2, distance])
                                    fw.write(f'\n {atom_name_1[0]}: {atom_ID:>3} \t {atom_name_2[0]}: {atom_ID_2:>3} \t Distance: {round(float(distance), 6)} ')
                
                if minmax_stats:
                    Host_Secondary_Distances = np.array(Host_Secondary_Distances, dtype=np.float_)
                    minimum_dist = Host_Secondary_Distances[np.argmin(Host_Secondary_Distances[:,2])]
                    maximum_dist = Host_Secondary_Distances[np.argmax(Host_Secondary_Distances[:,2])]
                    fw.write(f'\nMin-dist ({atom_name_1[0]} {int(minimum_dist[0])} - {atom_name_2[0]} {int(minimum_dist[1])}): {minimum_dist[2]} \t'+
                             f'Max-dist ({atom_name_1[0]} {int(maximum_dist[0])} - {atom_name_2[0]} {int(maximum_dist[1])}): {maximum_dist[2]}')




def vmd_bond_plugin(atom_ref, atom_search, rcut):
    '''
       atom
    '''

    vmd_func = '''proc remove_long_bonds { max_length } {
    for { set i 0 } { $i < [ molinfo top get numatoms ] } { incr i } {
        set bead [ atomselect top "index $i" ]
        set bonds [ lindex [$bead getbonds] 0 ]
        if { [ llength bonds ] > 0 } {
            set bonds_new {}
            set xyz [ lindex [$bead get {x y z}] 0 ]
            foreach j $bonds {
                set bead_to [ atomselect top "index $j" ]
                set xyz_to [ lindex [$bead_to get {x y z}] 0 ]
                if { [ vecdist $xyz $xyz_to ] < $max_length } {
                    lappend bonds_new $j
                }
            }
            $bead setbonds [ list $bonds_new ]
            }
        }
    }
    '''
    fname = Directory+atom_ref+'-'+atom_search+'-bond.tcl'

    with open(fname, 'w') as fw:
        
        atom_Data = Trajectory(filename=fileTraj)
        
        Atom_ID = 0
        for i, value in enumerate(atom_Data.coordinates[0]):

            if  value[0] == atom_ref:
                coord_atom_1 = np.array(value[1:], dtype=np.float_)
                
                Atom_ID_secondary = 0
                for j, value2 in enumerate(atom_Data.coordinates[0]):
   
                    if value2[0] == atom_search:
                        coord_atom_2 = np.array(value2[1:], dtype=np.float_)
   
                        _, distance = displacement(coord_atom_1, coord_atom_2)
                        
                        if distance <= rcut:
                            fw.write(f'topo addbond {Atom_ID} {Atom_ID_secondary} \n ')
                    Atom_ID_secondary += 1
            Atom_ID += 1
        fw.write('\n')
        fw.write(vmd_func)

