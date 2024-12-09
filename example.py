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
  H    6.6181727    4.9759502    3.2435439
  C    6.9813588    5.1692796    2.2424478
  C    6.0900782    5.1799396    1.1655316
  H    5.0370046    4.9947794    1.3336513
  C    6.5577724    5.4296108   -0.1288189
  H    5.8586015    5.4355043   -0.9556332
  C    7.9225445    5.6702836   -0.3518605
  N    8.3978682    5.9233003   -1.6620530
  H    7.7504781    5.9337213   -2.4638193
  H    9.3984007    6.1008534   -1.8336957
  C    8.8126797    5.6583033    0.7334006
  H    9.8683363    5.8421714    0.5775875
  C    8.3416806    5.4082969    2.0264872
  H    9.0310557    5.3998559    2.8608752
"""

##### Run electronic structure for molecule 1

#Make Molecule Object
molA = gto.M(atom=ala1, basis='sto-3g')

#Make SCF Object, Diagonalize Fock Matrix
mfA = scf.RKS(molA).run(xc='b3lyp')
print("done with SCF - Molecule A")
moA = mfA.mo_coeff #MO Coefficients
o_A = moA[:,mfA.mo_occ!=0] #occupied orbitals
v_A = moA[:,mfA.mo_occ==0] #virtual orbitals
tdA = mfA.TDA().run() #Do TDDFT-TDA

##### Run electronic structure for molecule 2

#Make Molecule Object
molB = gto.M(atom=ala2, basis='sto-3g') 

#Make SCF Object, Diagonalize Fock Matrix
mfB = scf.RKS(molB).run(xc='b3lyp')
print("done with SCF - Molecule B")
moB = mfB.mo_coeff #MO Coefficients
o_B = moB[:,mfB.mo_occ!=0] #occupied orbitals
v_B = moB[:,mfB.mo_occ==0] #virtual orbitals
tdB = mfB.TDA().run() #Do TDDFT-TDA

#%%
##### Extract Information for given excited state

state_id = 0  # first excited state

#excitation energies
e_A = tdA.e[state_id]
e_B = tdB.e[state_id]

print("Excitation energy Molecule A - ",e_A)
print("Excitation energy Molecule B - ",e_B)

#%%
""" Process Transition Density """

# The CIS coeffcients, shape [nocc,nvirt]
cis_A = tdA.xy[state_id][0]
cis_A *= 1. / np.linalg.norm(cis_A)

cis_B = tdB.xy[state_id][0]
cis_B *= 1. / np.linalg.norm(cis_B)

#Calculate Ground to Excited State density matrix
tdmA = np.sqrt(2) * o_A.dot(cis_A).dot(v_A.T)
tdmB = np.sqrt(2) * o_B.dot(cis_B).dot(v_B.T)

# Mullikan population analysis using transition density matrices
popAm,chrgAm = pop_mulliken(molA,tdmA)
popAm,chrgBm = pop_mulliken(molB,tdmB)

# Lowdin population analysis
popAl,chrgAl = pop_lowdin(molA,tdmA)
popBl,chrgBl = pop_lowdin(molB,tdmB)

# Natural population analysis
C_A = lo.orth_ao(mfA, 'nao')
moA_nao = np.linalg.solve(C_A, mfA.mo_coeff)
o_A_nao = moA_nao[:,mfA.mo_occ!=0] 
v_A_nao = moA_nao[:,mfA.mo_occ==0] 
tdmA_nao = np.sqrt(2) * o_A_nao.dot(cis_A).dot(v_A_nao.T)

popAnat,chrgAnat = pop_mulliken(molA, tdmA_nao, np.eye(molA.nao_nr()))

C_B = lo.orth_ao(mfB, 'nao')
moB_nao = np.linalg.solve(C_B, mfB.mo_coeff)
o_B_nao = moB_nao[:,mfB.mo_occ!=0] 
v_B_nao = moB_nao[:,mfB.mo_occ==0] 
tdmB_nao = np.sqrt(2) * o_B_nao.dot(cis_B).dot(v_B_nao.T)

popBnat,chrgBnat = pop_mulliken(molB, tdmB_nao, np.eye(molB.nao_nr()))

    
#%%
""" Calculate and Compare Couplings """

# Couplings standard method, note this can become very expensive if using a larger basis set
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

# Couplings with Mulliken transition charge projection
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

print("\n Couplings (kJ/mol)")
print("c_Full = %4.4f, cFull_eff = %4.4f, c_Coulomb = %4.4f, c_Exchange = %4.4f, c_Mul = %4.4f, c_Low = %4.4f, c_Nat = %4.4f\n"%(
    (cJ1 - cK1), (cJ2 - cK2), cJ2, cK2, Jq_mul, Jq_low, Jq_nat))

print("Time (s)")
print("t Full = %0.1f, t Full_eff = %0.6f, t Mul = %0.6f, t Low = %0.6f, t Nat= %0.6f\n"%(
    (t1,t2,t_mul,t_low,t_nat)))