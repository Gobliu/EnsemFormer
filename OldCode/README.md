# CycPeptMP3D

in CPMP/train_pampa.py, line 101: change gpu = "cuda:2" to "cuda:0
in CPMP/predict_pampa.py, line 60: change gpu = "cuda:2" to "cuda:0

in /egnn/qm9/data/prepare/qm9.py", line 216:
    charge_counts = {z: np.zeros(len(charges), dtype=np.int)
    to
    charge_counts = {z: np.zeros(len(charges), dtype=int)
