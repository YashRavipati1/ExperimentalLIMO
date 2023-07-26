import pandas as pd
from utils import smiles_to_one_hots
import os
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

kd_to_delta_g = lambda x: 0.00198720425864083 * 298.15 * math.log(x)

# Read the TSV file
df = pd.read_csv("D9AEC8E22C493F03A1087F8AF468CFD6ki.tsv", sep="\t")

# Access the "Age" column
smiles = df["Ligand SMILES"]
affinity = df["Ki (nM)"]


x = []
y = []
for i in range(len(smiles)):
    one_hots = smiles_to_one_hots(smiles[i])
    x.append(one_hots)
    y.append(kd_to_delta_g(affinity[i]))
print(x)
