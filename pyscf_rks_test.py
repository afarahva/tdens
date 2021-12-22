import numpy as np
import h5py
import scipy.linalg
from pyscf import gto, scf, tdscf, lib
import os
import sys
import time as time
from optparse import OptionParser

print("Num of Threads = "+str(lib.num_threads()))

"""
pop_dft_testing.py

Author: Ardavan Farahvash / Qiming Sun
Willard Group, MIT

Testing Various Atomic Population Methods Using DFT and PySCF
"""
import numpy as np
import h5py
import scipy.linalg
from pyscf import gto, scf, tdscf, lib, dft, lo
from functools import reduce

#coordinates of T2 molecule 1
xyz = '''
  H        0.00000       0.00000        0.00000
  H        1.00000       0.00000        0.00000
  '''


print("Num of Threads Before Calculation = "+str(lib.num_threads()))

#gto.M, gaussian type orbital molecule object
#scf.RKS, restricted kohn sham type Fock Matrix
#scf.RKS.TDA, TDDFT Tamm-Dancoff Approximation, Object  

#Make Molecule Object
molA = gto.M(atom=xyz, basis='6-31+g(d)') 
#Make SCF Object, Diagonalize Fock Matrix
mfA = scf.RKS(molA).run(xc='b3lyp') #camb3lyp causes problems with TDDFT, may need to reinstall

print("Num of Threads Right After DFT Calculation = "+str(lib.num_threads()))


moA = mfA.mo_coeff #MO Coefficients
o_A = moA[:,mfA.mo_occ!=0] #occupied orbitals
v_A = moA[:,mfA.mo_occ==0] #virtual orbitals
tdA = mfA.TDA().run() #Do TDDFT-TDA

print("Num of Threads Right AFTER TDDFT Calculation = "+str(lib.num_threads()))
