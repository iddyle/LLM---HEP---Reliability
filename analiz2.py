import uproot
import numpy as np
import matplotlib.pyplot as plt

# ROOT dosyasını 
f = uproot.open("JetNtuple_RunIISummer16_13TeV_MC_1.root")
tree = f["AK4jets/jetTree;3"]

# Verileri oku
branches = [
    "isPhysUDS",
    "isPhysG",
    "QG_ptD",
    "QG_axis2",
    "QG_mult"
]

data = tree.arrays(branches, library="np")

# Quark ve gluon maskeleri
quark_mask = data["isPhysUDS"] == 1
gluon_mask = data["isPhysG"] == 1

# Sınıflandırma
print("Quark sayısı:", np.sum(quark_mask))
print("Gluon sayısı:", np.sum(gluon_mask))

# Karşılaştırılacak değişkenler
features = ["QG_ptD", "QG_axis2", "QG_mult"]

for feature in features:

    quark_values = data[feature][quark_mask]
    gluon_values = data[feature][gluon_mask]

    print("\n", feature)
    print("Quark mean:", np.mean(quark_values))
    print("Gluon mean:", np.mean(gluon_values))

    plt.figure(figsize=(8, 5))

    plt.hist(
        quark_values,
        bins=40,
        density=True,
        alpha=0.5,
        label="UDS Quark"
    )

    plt.hist(
        gluon_values,
        bins=40,
        density=True,
        alpha=0.5,
        label="Gluon"
    )

    plt.xlabel(feature)
    plt.ylabel("Normalized density")
    plt.title(f"Quark vs Gluon: {feature}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{feature}_comparison.png")
    plt.show()