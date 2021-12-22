"""
transition_monopole_coupling.py

Author: Ardavan Farahvash / Qiming Sun
Willard Group, MIT

Lowdin transition charge monopole coupling Using DFT and PySCF
"""
import numpy as np
import h5py
import scipy.linalg
from pyscf import gto, scf, tdscf, lib, dft, lo
from functools import reduce

#coordinates of T2 molecule 1
t2_xyz1 = '''
  C       -1.05798        1.59414        0.00000
  C       -0.67546        2.95945        0.00000
  C        0.67546        3.13105        0.00000
  H       -2.08662        1.26459        0.00000
  H       -1.37857        3.77886        0.00000
  H        1.23616        4.04990        0.00000
  S        1.50576        1.61708        0.00000
  C        0.00323        0.72326        0.00000
  C       -0.00323       -0.72326        0.00000
  C        1.05798       -1.59414        0.00000
  S       -1.50576       -1.61708        0.00000
  C        0.67546       -2.95945        0.00000
  H        2.08662       -1.26459        0.00000
  C       -0.67546       -3.13105        0.00000
  H        1.37857       -3.77886        0.00000
  H       -1.23616       -4.04990        0.00000
  '''

# #coordinates of T2 molecule 2
t2_xyz2 = '''
C        2.047169       3.085793       -1.999515
C        3.092873       3.833562       -2.756783
C        4.031010       3.988056       -2.286310
C        3.670612       3.624355       -0.970658
S        2.153355       3.333692       -0.444138
C        6.045063       3.465878        1.744032
C        5.005070       3.555418        1.778312
C        4.000027       3.818073        0.864348
C        4.296131       3.917206        0.069971
S        5.612858       3.825866        0.750909
H        1.268799       3.791119       -2.250618
H        3.174225       3.340241       -3.707078
H        4.787903       3.791420       -3.031181
H        7.061745       3.405055        2.103271
H        4.805805       2.548391        2.113856
H        3.199448       3.098361        0.777827
  '''

#gto.M, gaussian type orbital molecule object
#scf.RKS, restricted kohn sham type Fock Matrix
#scf.RKS.TDA, TDDFT Tamm-Dancoff Approximation, Object  

#Make Molecule Object
molA = gto.M(atom=t2_xyz1, basis='6-31+g(d)') 
#Make SCF Object, Diagonalize Fock Matrix
mfA = scf.RKS(molA).run(xc='b3lyp') #camb3lyp causes problems with TDDFT, may need to reinstall
print("done with SCF - Molecule A")
moA = mfA.mo_coeff #MO Coefficients
o_A = moA[:,mfA.mo_occ!=0] #occupied orbitals
v_A = moA[:,mfA.mo_occ==0] #virtual orbitals
tdA = mfA.TDA().run() #Do TDDFT-TDA


#Make Molecule Object
molB = gto.M(atom=t2_xyz2, basis='6-31+g(d)') 
#Make SCF Object, Diagonalize Fock Matrix
mfB = scf.RKS(molB).run(xc='b3lyp') #camb3lyp causes problems with TDDFT 
print("done with SCF - Molecule B")
moB = mfB.mo_coeff #MO Coefficients
o_B = moB[:,mfB.mo_occ!=0] #occupied orbitals
v_B = moB[:,mfB.mo_occ==0] #virtual orbitals
tdB = mfB.TDA().run() #Do TDDFT-TDA

state_id = 0  # first excited state
#excitation energies
e_A = tdA.e[state_id]
e_A = tdB.e[state_id]

# The CIS coeffcients, shape [nocc,nvirt]
#Index 0 ~ X matrix/CIS coefficients, Index Y ~ Deexcitation Coefficients
cis_A = tdA.xy[state_id][0] 
cis_B = tdB.xy[state_id][0]

#%%
#Calculate Transitio

def get_ovlp(self, mol=None):
    if mol is None: mol = self.mol
    return get_ovlp(mol)

def td_chrg_lowdin(mol, dm, s):
    """
    Calculates Lowdin Transition Partial Charges
    
    Parameters
    ----------
    mol. PySCF Molecule Object
    dm. Numpy Array. Transition Density Matrix in Atomic Orbital Basis
    s. Numpy Array. Atomic Orbital Overlap basis
    
    Returns
    -------
    pop. Numpy Array. Population in each orbital.
    chg. Numpy Array. Charge on each atom.
    """

    
    U,s_diag,_ = np.linalg.svd(s,hermitian=True)
    S_half = U.dot(np.diag(s_diag**(0.5))).dot(U.T)
    
    pop = np.einsum('ij,jk,ki->i',S_half, dm, S_half)

    print(' ** Lowdin atomic charges  **')
    chg = np.zeros(mol.natm)
    for i, s in enumerate(mol.ao_labels(fmt=None)):
        chg[s[0]] += pop[i]
        
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        print('charge of  %d%s =   %10.5f'%(ia, symb, chg[ia]))
    
    return pop, chg



#Calculate Transition Density Matrices in Atomic Orbital Basis
tdmA = np.sqrt(2) * o_A.dot(cis_A).dot(v_A.T)
tdmB = np.sqrt(2) * o_B.dot(cis_B).dot(v_B.T)

#Lowdin population analysis
popAl,chrgAl = td_chrg_lowdin(molA,tdmA,scf.hf.get_ovlp(molA))
popBl,chrgBl = td_chrg_lowdin(molB,tdmB,scf.hf.get_ovlp(molB))
    
#%%
#Exact 2 Molecule Coupling Calculations

def jk_ints_standard(molA,molB,mfA,mfB,cisA,cisB):
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
    cJ = np.einsum('iabj,ia,jb->', eri_iabj, cis_A, cis_B)
    cK = np.einsum('ijba,ia,jb->', eri_ijba, cis_A, cis_B)
    
    return cJ, cK

#More Efficient BlackBox Implementation
def jk_ints_eff(molA, molB, tdmA, tdmB, calcK=False):
    """
    A more-efficient version of two-molecule JK integrals.
    This implementation is a bit blackbox and relies and calculation the HF
    potential before trying to calculate couplings. 

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
        
    if calcK==True:
        # Exchange integrals
        with lib.temporary_env(vhfopt._this.contents,
                               fprescreen=_vhf._fpointer('CVHFnrs8_vk_prescreen')):
            shls_slice = (0        , molA.nbas , molA.nbas, molAB.nbas,
                          molA.nbas, molAB.nbas, 0        , molA.nbas)  # ABBA
            vK = jk.get_jk(molAB, tdmB, 'ijkl,jk->il', shls_slice=shls_slice,
                           vhfopt=vhfopt, aosym='s1', hermi=0)
            cK = np.einsum('ia,ia->', vK, tdmA)
            
        return cJ, cK
    
    else: 
        return cJ

#cJ1,cK1 = jk_ints_standard(molA,molB,mfA,mfB,cis_A,cis_B) * 2625.50

cJ2,cK2 = jk_ints_eff(molA,molB,tdmA,tdmB,calcK=True)
cJ2 = cJ2 * 2625.50 #converts from Hartree to eV
cK2 = cK2 * 2625.50

from scipy.spatial.distance import cdist,pdist

Jq_low = np.sum( np.outer(chrgAl,chrgBl)/cdist(molA.atom_coords(),molB.atom_coords()) ) * 2625.50

#you can compare the exact coupling cJ2 to the transition charge coupling Jq_low
