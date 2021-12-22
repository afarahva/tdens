"""
tdens
============

file: tdens.py
author: Ardavan Farahvash (MIT)

description: 
Functions to project transition density (in AO basis) to transition monopole
charge using PySCF

Please cite: https://doi.org/10.1063/5.0016009
"""
import numpy as np
import scipy.linalg
from pyscf import gto, scf, tdscf, lib, dft, lo
from functools import reduce

def get_ovlp(self, mol=None):
    if mol is None: mol = self.mol
    return get_ovlp(mol)

def pop_mulliken(mol, dm, s=None):
    """
    Mulliken population analysis
    
    P_{\mu} = Tr[P_{\mu\nu}S_{\mu\nu}]
    
    Where P_{\mu} is the population in each atomic orbital. 
    
    q_{a} = \sum_{\mu \in a} P_\mu
    
    q_{a} is the partial charge of atom a
    
    Parameters
    ----------
    mol: PyScf Mol Object.
    dm: Numpy Array. Density Matrix (AO Basis)
    s: Numpy Array. Overlap Matrix

    Returns
    -------
    pop. Numpy Array. Population in each atomic orbital. 
    chg. Numpy Array. Charge on each atom. 
    """
    if np.all(s == None):
        s = scf.hf.get_ovlp(mol)
        
    pop = np.einsum('ij,ji->i', dm, s)

    chg = np.zeros(mol.natm)
    for i, s in enumerate(mol.ao_labels(fmt=None)):
        chg[s[0]] += pop[i]
    
    return pop, chg

def pop_lowdin(mol, dm):
    """
    Lowdin population analysis
    
    Equivalent to Mullikan population analysis, but using Lowdin symmetrically
    orthoganlized AO as the basis. 
    
    P_{\mu} = Tr[S^{1/2} P_{\mu\nu} S^{1/2}]
    
    Where P_{\mu} is the population in each atomic orbital. 
    
    q_{a} = \sum_{\mu \in a} P_\mu
    
    q_{a} is the partial charge of atom a
    
    Parameters
    ----------
    mol: PyScf Mol Object.
    dm: Numpy Array. Density Matrix (AO Basis)

    Returns
    -------
    pop. Numpy Array. Population in each atomic orbital. 
    chg. Numpy Array. Charge on each atom. 
    """
    s = scf.hf.get_ovlp(mol)
    U,s_diag,_ = np.linalg.svd(s,hermitian=True)
    S_half = U.dot(np.diag(s_diag**(0.5))).dot(U.T)
    
    pop = np.einsum('ij,jk,ki->i',S_half, dm, S_half)
    
    chg = np.zeros(mol.natm)
    for i, s in enumerate(mol.ao_labels(fmt=None)):
        chg[s[0]] += pop[i]
        
    return pop, chg

def pop_minao(mol,dm):
    """
    Mulliken population analysis in minimal atomic basis set.
    
    Projects atomic orbitals into minimal basis set, and then calculates population
    
    Parameters
    ----------
    mol: PyScf Mol Object.
    dm: Numpy Array. Density Matrix (AO Basis)

    Returns
    -------
    popmin. Numpy Array. Population in each atomic orbital. 
    chgmin. Numpy Array. Charge on each atom. 
    """
    
    mol_minao = gto.M(atom=mol.atom, basis="minao", unit="Angstrom")
    dm_minao = scf.addons.project_dm_nr2nr(mol, dm, mol_minao)
    popmin,chgmin = pop_mulliken(mol_minao,dm_minao)
    
    return popmin,chgmin
    
def pop_natural(mol,dm):
    """
    Mulliken population analysis based on natural atomic orbitals
    
    Natural atomic orbitals are chosen such that they diagonalize the molecular
    Fock matrix (they are like MO's in this sense). 
    
    Parameters
    ----------
    mol: PyScf Mol Object.
    dm: Numpy Array. Density Matrix (AO Basis)

    Returns
    -------
    popnat. Numpy Array. Population in each atomic orbital. 
    chgnat. Numpy Array. Charge on each atom. 
    """
    #Fock Matrix
    mf = scf.RKS(mol).run(xc='b3lyp') #camb3lyp causes problems with TDDFT, may need to reinstall
    
    #AO Coefficients in natural atomic orbital basis
    C = lo.orth_ao(mf, 'nao')
    Cinv = np.linalg.inv(C)
    
    #density matrix in NAO basis
    dm_natao = Cinv.dot(dm).dot(Cinv.T)

    #Mullikan Pops using B3LYP NAO (Natural Population Analysis)
    popnat,chgnat = pop_mulliken(mol, dm_natao, s=np.eye(mol.nao_nr()) )
    return popnat, chgnat

def exc_density_matrix(tamm_danc, state_id):
    """
    Generate  Excited State 1-RDM From TDDFT/TDSCF-Tamm-Dancoff Object

    Parameters
    ----------
    tamm_danc : PySCF Obj. TDDFT/TDSCF-Tamm-Dancoff Object. 
    state_id : Excited State to analyze, 0 ~ first excited state

    Returns
    -------
    dm : Numpy Array. Density Matrix in Molecular Orbital Basis.
    dm_ao : Numpy Array. Density Matric in Atomic Orbital Basis.
    """

    cis_t1 = tamm_danc.xy[state_id][0]
    
    #CIS Coefficient Matrices (-) DD^T ~ Sum of Virtual Indicesx, D^TD ~ Sum of Occupied Indices
    dm_oo =-np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
    dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())

    # The ground state density matrix in MO Basis
    # every occupied MO has an occupancy of 2, all other have occupancy of 0
    mf = tamm_danc._scf     #extract Fock Matrix from TDA object
    dm = np.diag(mf.mo_occ) #ground-state density matrix in MO Basis

    # Add CIS contribution
    # Due to CIS being a linear combination of singley-excited determinants
    # We need to take into account off-diagonal contributions from virtual orbitals
    nocc = cis_t1.shape[0]
    dm[:nocc,:nocc] += dm_oo * 2
    dm[nocc:,nocc:] += dm_vv * 2

    # Transform density matrix to AO basis
    mo = mf.mo_coeff
    dm_ao = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
    return dm, dm_ao

if __name__ == "__main__":
    pass
