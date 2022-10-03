import json
import os
cwd = os.path.dirname(__file__)

with open(cwd+'/'+'periodic_table.json', 'r') as jsonData:
    periodicData = json.load(jsonData)
    jsonData.close()


def atomic_no(atom_symbol):
    ''' Returns the atomic number of passed atom_symbol.

    Parameters:
    ---------------------------------------------------
    atom_symbol  : Atomic symbol (string)

    Return:
    ---------------------------------------------------
    atomic_num   : atomic number (float)
    '''
    assert isinstance(atom_symbol, str)
    atomic_num = periodicData[atom_symbol]['Atomic no']    
    return atomic_num


def atomic_symbol(atomic_no):
    ''' Returns the atomic symbol of atomic number passed.

    Parameters:
    -----------------------------------------------------
    atomic_no   : Atomic number (int)
    
    Return:
    -----------------------------------------------------
    atomic_sym  : Atomic symbol (str)
    '''
    atomic_no = int(float(atomic_no))
    atomic_sym = [i for i in periodicData if periodicData[i]['Atomic no']==atomic_no][0]
    return atomic_sym

def get_atomic_IDs(coordinates, atom):
    '''Gives the list containing IDs of {atom}
       
    Parameters:
    --------------------------------------------------
    coordinates : x,y & z coordinates of atomic configuration
    atom : Atmomic number or symbol

    Return:
    --------------------------------------------------
    atom_list : list of all the ids of of atom   
    '''
    
    if isinstance(atom, str):
        atom = atomic_no(atom)
    atom_ID_list = []

    for atom_ID, value in enumerate(coordinates):
        if value[0] == atom:
            atom_ID_list.append(atom_ID)

    return atom_ID_list


def atomic_mass(atom, gram=False):
    if not isinstance(atom, str):
        atom = atomic_symbol(atom)
    
    mass =  periodicData[atom]['Atomic mass']

    # 1 u = 1.6605402E-24 g
    if gram:
        mass = mass * 1.6605402

    return mass


