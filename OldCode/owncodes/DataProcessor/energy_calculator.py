from rdkit import Chem
from rdkit.Chem import AllChem

# 1. Load molecule and add hydrogens
mol = Chem.MolFromPDBFile("../../../Data/For_JG/ff_uff_ig_False/2015_Wang_1045.pdb")

# MMFF energy (gas phase)
mp = AllChem.MMFFGetMoleculeProperties(mol)
ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
energy = ff.CalcEnergy()
print(f"MMFF Energy : {energy:.2f} kcal/mol")
