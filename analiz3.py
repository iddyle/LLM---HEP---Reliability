
import uproot
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu

f = uproot.open("JetNtuple_RunIISummer16_13TeV_MC_1.root")
tree = f["AK4jets/jetTree;3"]

branches = [
    "isPhysUDS",
    "isPhysG",
    "QG_ptD",
    "QG_axis2",
    "QG_mult"
]

data = tree.arrays(branches, library="np")

quark = data["isPhysUDS"] == 1
gluon = data["isPhysG"] == 1

features = ["QG_ptD", "QG_axis2", "QG_mult"]

print("=== QUARK-GLUON DISCRIMINATION ===")

for feature in features:

    q = data[feature][quark]
    g = data[feature][gluon]

    # ROC-AUC: Gluon score > Quark score
    result = mannwhitneyu(g, q, alternative="two-sided")
    auc = result.statistic / (len(g) * len(q))

    discrimination_auc = max(auc, 1 - auc)

    # Kolmogorov-Smirnov test
    ks = ks_2samp(q, g)

    print(f"\n{feature}")
    print("Raw AUC:", auc)
    print("Discrimination AUC:", discrimination_auc)
    print("KS statistic:", ks.statistic)
    print("KS p-value:", ks.pvalue)
