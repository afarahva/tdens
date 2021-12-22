"""
tdens
============

file: coupling.py
author: Ardavan Farahvash (MIT) with significant contributions from Dr. Qiming
Sun (Caltech)

description: 
Functions to calculate excitonic coupling integral using PySCF

Please cite: https://doi.org/10.1063/5.0016009
"""

import numpy as np
import scipy.linalg
from pyscf import gto, scf, tdscf, lib, dft, lo
from functools import reduce
from scipy.spatial.distance import cdist

def jk_ints_standard(molA,molB,mfA,mfB,cisA,cisB, calcK=False):
    """
    A standard implementation of two-molecule JK integrals.
    This implementation is very memory intensive and very computationally heavy
    But, it is straightforward.
    

    Parameters
    ----------
    molA/molB : PySCF Mol Obj. Molecule A and Molecule B.
    mfA/mfB : PySCF SCF Obj. Fock Matrix for Molecule A and Molecule B. 
    cisA/cisB : Numpy Array. CIS coefficients (nocc x nvirt)

    Returns
    -------
    cJ ~ Coulomb Coupling
    cK ~ Exchange Coupling
    
    V_{ab} = 2J - K
    """
    
    #Extract MO Coefficients
    moA = mfA.mo_coeff #MO Coefficients
    o_A = moA[:,mfA.mo_occ!=0] #occupied orbitals
    v_A = moA[:,mfA.mo_occ==0] #virtual orbitals
    moB = mfB.mo_coeff #MO Coefficients
    o_B = moB[:,mfB.mo_occ!=0] #occupied orbitals
    v_B = moB[:,mfB.mo_occ==0] #virtual orbitals    
    
    #Create 2 molecule object
    molAB = molA + molB 
    naoA = molA.nao #of atomic orbitals in molecule A
    
    #two electron repulsion integral for all electrons in 2 molecule system
    eri = molAB.intor('int2e')
    
    #AA-BB block of two electron integral
    eri_AABB = eri[:naoA,:naoA,naoA:,naoA:] 
    
    #AB-BA block of two electron integral
    eri_ABBA = eri[:naoA,naoA:,naoA:,:naoA]
    
    # Transform integrals to from AO to MO basis
    eri_iabj = lib.einsum('pqrs,pi,qa,rb,sj->iabj', eri_AABB, o_A, v_A, v_B, o_B)
    eri_ijba = lib.einsum('pqrs,pi,qj,rb,sa->ijba', eri_ABBA, o_A, o_B, v_B, v_A)
    
    # J-type coupling and K-type coupling
    cJ = np.einsum('iabj,ia,jb->', eri_iabj, cisA, cisB)
    
    if calcK == False:
        return cJ
    else:
        cK = np.einsum('ijba,ia,jb->', eri_ijba, cisA, cisB)
        return cJ, cK

#More Efficient BlackBox Implementation
def jk_ints_eff(molA, molB, tdmA, tdmB, calcK=False):
    """
    A more-efficient version of two-molecule JK integrals.
    
    Uses Density Fitting in Auxilary Basis to Calculate Couplings

    Credit goes to Qiming Sun for this code

    Parameters
    ----------
    molA/molB : PySCF Mol Obj. Molecule A and Molecule B.
    tdmA/tdmB : Numpy Array. Transiiton density Matrix

    Returns
    -------
    cJ ~ Coulomb Coupling
    cK ~ Exchange Coupling
    
    V_{ab} = 2J - K
    """
    
    from pyscf.scf import jk, _vhf
    naoA = molA.nao
    naoB = molB.nao
    assert(tdmA.shape == (naoA, naoA))
    assert(tdmB.shape == (naoB, naoB))

    molAB = molA + molB
    
    #vhf = Hartree Fock Potential
    vhfopt = _vhf.VHFOpt(molAB, 'int2e', 'CVHFnrs8_prescreen',
                         'CVHFsetnr_direct_scf',
                         'CVHFsetnr_direct_scf_dm')
    dmAB = scipy.linalg.block_diag(tdmA, tdmB)
    #### Initialization for AO-direct JK builder
    # The prescreen function CVHFnrs8_prescreen indexes q_cond and dm_cond
    # over the entire basis.  "set_dm" in function jk.get_jk/direct_bindm only
    # creates a subblock of dm_cond which is not compatible with
    # CVHFnrs8_prescreen.
    vhfopt.set_dm(dmAB, molAB._atm, molAB._bas, molAB._env)
    # Then skip the "set_dm" initialization in function jk.get_jk/direct_bindm.
    vhfopt._dmcondname = None
    ####

    # Coulomb integrals
    with lib.temporary_env(vhfopt._this.contents,
                           fprescreen=_vhf._fpointer('CVHFnrs8_vj_prescreen')):
        shls_slice = (0        , molA.nbas , 0        , molA.nbas,
                      molA.nbas, molAB.nbas, molA.nbas, molAB.nbas)  # AABB
        vJ = jk.get_jk(molAB, tdmB, 'ijkl,lk->s2ij', shls_slice=shls_slice,
                       vhfopt=vhfopt, aosym='s4', hermi=1)
        cJ = np.einsum('ia,ia->', vJ, tdmA)
        
    if calcK==False:
        return cJ
    
    # Exchange integrals
    else:
        with lib.temporary_env(vhfopt._this.contents,
                               fprescreen=_vhf._fpointer('CVHFnrs8_vk_prescreen')):
            shls_slice = (0        , molA.nbas , molA.nbas, molAB.nbas,
                          molA.nbas, molAB.nbas, 0        , molA.nbas)  # ABBA
            vK = jk.get_jk(molAB, tdmB, 'ijkl,jk->il', shls_slice=shls_slice,
                           vhfopt=vhfopt, aosym='s1', hermi=0)
            cK = np.einsum('ia,ia->', vK, tdmA)
            
        return cJ, cK
    

def coupling_tdchg(chgA,chgB,coordsA,coordsB):
    """
    Parameters
    ----------
    chgA : Numpy Array. Charges on Atoms in Mol A
    chgB : Numpy Array. Charges on Atoms in Mol A
    coordsA : Numpy Array. Coordinates of Mol A (Bohr)
    coordsB : Numpy Array. Coordinates of Mol B (Bohr)

    Returns
    -------
    cJ : Float. Coupling (kJ/mol)
    """
    
    cJ = np.sum( np.outer(chgA,chgB)/cdist(coordsA,coordsB) )
    return cJ

if __name__ == "__main__":
    pass