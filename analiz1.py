import uproot
import numpy as np

f = uproot.open("JetNtuple_RunIISummer16_13TeV_MC_1.root")
tree = f["AK4jets/jetTree;3"]

# İhtiyacımız olan değişkenleri oku
branches = [
    "isPhysUDS",
    "isPhysG",
    "isPhysOther",
    "QG_ptD",
    "QG_axis2",
    "QG_mult"
]

data = tree.arrays(branches, library="np")

# Sınıf etiketlerini incele
uds = data["isPhysUDS"]
gluon = data["isPhysG"]
other = data["isPhysOther"]

print("QUARK-GLUON SINIF DAĞILIMI")
print("-------------------------")
print("UDS quark:", np.sum(uds))
print("Gluon:", np.sum(gluon))
print("Other:", np.sum(other))

print("\nTOPLAM")
print(len(uds))

# Değişkenlerin istatistikleri
print("\nDEĞİŞKEN İSTATİSTİKLERİ")
for name in ["QG_ptD", "QG_axis2", "QG_mult"]:
    values = data[name]
    print(f"\n{name}")
    print("Mean:", np.nanmean(values))
    print("Std:", np.nanstd(values))
    print("Min:", np.nanmin(values))
    print("Max:", np.nanmax(values))
    print("NaN:", np.isnan(values).sum())