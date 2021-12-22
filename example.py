"""
tdens
============

file: example.py
author: Ardavan Farahvash (MIT)

description: 
Example script to calculate transition densities and exciton couplings

Please cite: https://doi.org/10.1063/5.0016009
"""

import numpy as np
from pyscf import gto, scf, tdscf, lib, dft, lo
from time import time

from tdens import *
from coupling import *

# coordinates of alanine molecule 1
ala1 = \
"""
  H    0.1943255    1.0032856    2.5356066
  C   -0.3596931    1.0137933    1.6058611
  C   -1.7521481    0.8909298    1.6205948
  H   -2.2754784    0.7853375    2.5621298
  C   -2.4696264    0.9043176    0.4199855
  H   -3.5479438    0.8084300    0.4430776
  C   -1.7954144    1.0410213   -0.8035442
  N   -2.5204902    1.0547696   -2.0203556
  H   -3.5471291    0.9649865   -2.0221873
  H   -2.0325467    1.1547631   -2.9225553
  C   -0.3971725    1.1639953   -0.8120179
  H    0.1373472    1.2701957   -1.7477028
  C    0.3172682    1.1502270    0.3903971
  H    1.3953982    1.2452971    0.3799184
"""

# coordinates of alanine molecule 2
ala2 = \
"""
  H    4.6181727    0.9759502    3.2435439
  C    4.9813588    1.1692796    2.2424478
  C    4.0900782    1.1799396    1.1655316
  H    3.0370046    0.9947794    1.3336513
  C    4.5577724    1.4296108   -0.1288189
  H    3.8586015    1.4355043   -0.9556332
  C    5.9225445    1.6702836   -0.3518605
  N    6.3978682    1.9233003   -1.6620530
  H    5.7504781    1.9337213   -2.4638193
  H    7.3984007    2.1008534   -1.8336957
  C    6.8126797    1.6583033    0.7334006
  H    7.8683363    1.8421714    0.5775875
  C    6.3416806    1.4082969    2.0264872
  H    7.0310557    1.3998559    2.8608752
"""

""" Run Electronic Structure Calculations """
#gto.M, gaussian type orbital molecule object
#scf.RKS, restricted kohn sham type Fock Matrix
#scf.RKS.TDA, TDDFT Tamm-Dancoff Approximation, Object  

##### Run electronic structure for molecule 1

#Make Molecule Object
molA = gto.M(atom=ala1, basis='6-31+g(d)')

#Make SCF Object, Diagonalize Fock Matrix
mfA = scf.RKS(molA).run(xc='b3lyp')
print("done with SCF - Molecule A")
moA = mfA.mo_coeff #MO Coefficients
o_A = moA[:,mfA.mo_occ!=0] #occupied orbitals
v_A = moA[:,mfA.mo_occ==0] #virtual orbitals
tdA = mfA.TDA().run() #Do TDDFT-TDA

##### Run electronic structure for molecule 2

#Make Molecule Object
molB = gto.M(atom=ala2, basis='6-31+g(d)') 

#Make SCF Object, Diagonalize Fock Matrix
mfB = scf.RKS(molB).run(xc='b3lyp')
print("done with SCF - Molecule B")
moB = mfB.mo_coeff #MO Coefficients
o_B = moB[:,mfB.mo_occ!=0] #occupied orbitals
v_B = moB[:,mfB.mo_occ==0] #virtual orbitals
tdB = mfB.TDA().run() #Do TDDFT-TDA

##### Extract Information for given excited state
state_id = 0  # first excited state

#excitation energies
e_A = tdA.e[state_id]
e_B = tdB.e[state_id]

#%%
""" Process Transition Density """

# The CIS coeffcients, shape [nocc,nvirt]
#Index 0 ~ X matrix/CIS coefficients, Index Y ~ Deexcitation Coefficients
cis_A = tdA.xy[state_id][0] 
cis_B = tdB.xy[state_id][0]

#Calculate Ground to Excited State density matrix
tdmA = np.sqrt(2) * o_A.dot(cis_A).dot(v_A.T)
tdmB = np.sqrt(2) * o_B.dot(cis_B).dot(v_B.T)

#Mullikan population analysis using transition density matrices
popAm,chrgAm = td_chrg_mulliken(molA,tdmA,scf.hf.get_ovlp(molA))
popAm,chrgBm = td_chrg_mulliken(molB,tdmB,scf.hf.get_ovlp(molB))

#Lowdin population analysis
popAl,chrgAl = td_chrg_lowdin(molA,tdmA,scf.hf.get_ovlp(molA))
popBl,chrgBl = td_chrg_lowdin(molB,tdmB,scf.hf.get_ovlp(molB))

#Natural Atomic Orbitals
C_A = lo.orth_ao(mfA, 'nao')
moA_nao = np.linalg.solve(C_A, mfA.mo_coeff)
o_A_nao = moA_nao[:,mfA.mo_occ!=0] 
v_A_nao = moA_nao[:,mfA.mo_occ==0] 
tdmA_nao = np.sqrt(2) * o_A_nao.dot(cis_A).dot(v_A_nao.T)

popAnat,chrgAnat = td_chrg_mulliken(molA, tdmA_nao, np.eye(molA.nao_nr()))

C_B = lo.orth_ao(mfB, 'nao')
moB_nao = np.linalg.solve(C_B, mfB.mo_coeff)
o_B_nao = moB_nao[:,mfB.mo_occ!=0] 
v_B_nao = moB_nao[:,mfB.mo_occ==0] 
tdmB_nao = np.sqrt(2) * o_B_nao.dot(cis_B).dot(v_B_nao.T)

popBnat,chrgBnat = td_chrg_mulliken(molB, tdmB_nao, np.eye(molB.nao_nr()))

    
#%%
""" Calculate and Compare Couplings """

# Couplings standard method
t = time()
cJ1,cK1 = jk_ints_standard(molA,molB,mfA,mfB,cis_A,cis_B,calcK=True)
cJ1 = cJ1 * 2625.50
cK1 = cK1 * 2625.50
t1 = time() - t

# Couplings with efficient density fitting method
t = time()
cJ2,cK2 = jk_ints_eff(molA,molB,tdmA,tdmB,calcK=True)
cJ2 = cJ2 * 2625.50
cK2 = cK2 * 2625.50
t2 = time() - t

# Couplings with Mullikan transition charge projection
t = time()
Jq_mul = coupling_tdchg(chrgAm,chrgBm,molA.atom_coords(),molB.atom_coords()) * 2625.50
t_mul = time() - t

# Couplings with Lowdin transition charge projection
t = time()
Jq_low = coupling_tdchg(chrgAl,chrgBl,molA.atom_coords(),molB.atom_coords()) * 2625.50
t_low = time() - t

# Couplings with Natural transition charge projection
t = time()
Jq_nat = coupling_tdchg(chrgAnat,chrgBnat,molA.atom_coords(),molB.atom_coords()) * 2625.50
t_nat = time() - t

print("Couplings (kJ/mol)")
print("cFull = %4.4f, cFull_eff = %4.4f, c_mul = %4.4f, c_low = %4.4f, c_mull= %4.4f",
      (cJ1 - cK1), (cJ2 - cK2), Jq_mul, Jq_low, Jq_nat)

print("Time (s)")
print("tFull = %d, tFull_eff = %d, t_mul = %d, t_low = %d, c_mull= %d",
      (t1,t2,t_mul,t_low,t_nat))