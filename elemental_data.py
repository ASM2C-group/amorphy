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
