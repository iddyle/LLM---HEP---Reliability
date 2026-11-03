
import json
from pathlib import Path

import numpy as np
import uproot
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


# --------------------------------------------------
# 1. DOSYA VE ANALİZ AYARLARI
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

ROOT_FILE = BASE_DIR / "JetNtuple_RunIISummer16_13TeV_MC_1.root"

TREE_NAME = "AK4jets/jetTree;3"

FEATURES = [
    "QG_ptD",
    "QG_axis2",
    "QG_mult"
]

CLASS_BRANCHES = [
    "isPhysUDS",
    "isPhysG",
    "isPhysOther"
]

OUTPUT_DIR = BASE_DIR / "sonuclar"


# --------------------------------------------------
# 2. ROOT VERİSİNİ OKUMA
# --------------------------------------------------

def load_root_data():

    if not ROOT_FILE.exists():
        raise FileNotFoundError(
            f"ROOT dosyası bulunamadı: {ROOT_FILE}"
        )

    required_branches = FEATURES + CLASS_BRANCHES

    with uproot.open(ROOT_FILE) as root_file:

        tree = root_file[TREE_NAME]

        available_branches = set(tree.keys())

        missing_branches = [
            branch
            for branch in required_branches
            if branch not in available_branches
        ]

        if missing_branches:
            raise ValueError(
                f"ROOT dosyasında eksik branch'ler: {missing_branches}"
            )

        arrays = tree.arrays(
            required_branches,
            library="np"
        )

    return arrays


# --------------------------------------------------
# 3. SINIF SEÇİMİ
# --------------------------------------------------

def select_classes(arrays):

    is_quark = np.asarray(
        arrays["isPhysUDS"]
    ).astype(bool)

    is_gluon = np.asarray(
        arrays["isPhysG"]
    ).astype(bool)

    is_other = np.asarray(
        arrays["isPhysOther"]
    ).astype(bool)

    # Sınıfların birbirini dışlayıp dışlamadığını kontrol et
    overlap = (
        is_quark.astype(int)
        + is_gluon.astype(int)
        + is_other.astype(int)
    )

    if np.any(overlap > 1):
        raise ValueError(
            "Sınıf etiketlerinde çakışma bulundu."
        )

    quark_count = int(np.sum(is_quark))
    gluon_count = int(np.sum(is_gluon))
    other_count = int(np.sum(is_other))

    return {
        "quark_mask": is_quark,
        "gluon_mask": is_gluon,
        "quark_count": quark_count,
        "gluon_count": gluon_count,
        "other_count": other_count
    }


# --------------------------------------------------
# 4. TEK BİR ÖZELLİĞİN ANALİZİ
# --------------------------------------------------

def analyze_feature(
    feature_name,
    feature_values,
    quark_mask,
    gluon_mask
):

    values = np.asarray(
        feature_values,
        dtype=float
    )

    # Sayısal olmayan ve sonsuz değerleri dışarıda bırak
    valid = np.isfinite(values)

    quark_values = values[
        quark_mask & valid
    ]

    gluon_values = values[
        gluon_mask & valid
    ]

    if len(quark_values) == 0 or len(gluon_values) == 0:
        raise ValueError(
            f"{feature_name}: Geçerli sınıf verisi bulunamadı."
        )

    # Ortalama değerler
    quark_mean = float(np.mean(quark_values))
    gluon_mean = float(np.mean(gluon_values))

    # Standart sapmalar
    quark_std = float(np.std(quark_values))
    gluon_std = float(np.std(gluon_values))

    # Gluon pozitif sınıf olarak tanımlanıyor.
    # Raw AUC, yüksek feature değerlerinin gluon olma
    # eğilimini ne kadar iyi sıraladığını gösterir.
    y_true = np.concatenate([
        np.zeros(len(quark_values)),
        np.ones(len(gluon_values))
    ])

    scores = np.concatenate([
        quark_values,
        gluon_values
    ])

    raw_auc = float(
        roc_auc_score(y_true, scores)
    )

    # Yön bağımsız ayrım gücü
    discrimination_auc = max(
        raw_auc,
        1.0 - raw_auc
    )

    # Kolmogorov-Smirnov testi
    ks_result = ks_2samp(
        quark_values,
        gluon_values
    )

    # Ortalama yönü
    if np.isclose(quark_mean, gluon_mean):
        mean_direction = "equal"
    elif gluon_mean > quark_mean:
        mean_direction = "gluon_higher"
    else:
        mean_direction = "quark_higher"

    # Raw AUC yönü
    if np.isclose(raw_auc, 0.5):
        raw_auc_direction = "equal_0.5"
    elif raw_auc > 0.5:
        raw_auc_direction = "above_0.5"
    else:
        raw_auc_direction = "below_0.5"

    return {
        "quark_mean": quark_mean,
        "gluon_mean": gluon_mean,
        "quark_std": quark_std,
        "gluon_std": gluon_std,
        "raw_auc": raw_auc,
        "discrimination_auc": discrimination_auc,
        "ks_statistic": float(ks_result.statistic),
        "ks_pvalue": float(ks_result.pvalue),
        "quark_valid_entries": int(len(quark_values)),
        "gluon_valid_entries": int(len(gluon_values)),
        "mean_direction": mean_direction,
        "raw_auc_direction": raw_auc_direction
    }


# --------------------------------------------------
# 5. TÜM ANALİZİ GERÇEKLEŞTİRME
# --------------------------------------------------

def run_reference_analysis():

    print("ROOT dosyası okunuyor...")

    arrays = load_root_data()

    print("Sınıflar seçiliyor...")

    classes = select_classes(arrays)

    quark_mask = classes["quark_mask"]
    gluon_mask = classes["gluon_mask"]

    results = {
        "dataset": ROOT_FILE.name,
        "tree": TREE_NAME,
        "total_entries": int(len(quark_mask)),
        "quark_count": classes["quark_count"],
        "gluon_count": classes["gluon_count"],
        "other_count": classes["other_count"],
        "selection": (
            "isPhysUDS == 1 and isPhysG == 1 "
            "are treated as separate classes; "
            "Other is excluded from binary comparison."
        ),
        "positive_class": "gluon",
        "features": {}
    }

    print("Fiziksel değişkenler analiz ediliyor...")

    for feature in FEATURES:

        results["features"][feature] = analyze_feature(
            feature,
            arrays[feature],
            quark_mask,
            gluon_mask
        )

    # Özellikleri ayrım gücüne göre sırala
    ranking = sorted(
        FEATURES,
        key=lambda name: results["features"][name][
            "discrimination_auc"
        ],
        reverse=True
    )

    results["ranking"] = ranking

    return results


# --------------------------------------------------
# 6. SONUÇLARI KAYDETME
# --------------------------------------------------

def save_results(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = OUTPUT_DIR / "referans_analiz.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nReferans sonuçlar kaydedildi:")
    print(output_file)


# --------------------------------------------------
# 7. SONUÇLARI EKRANA YAZDIRMA
# --------------------------------------------------

def print_results(results):

    print("\n" + "=" * 55)
    print("QCD JET REFERANS ANALİZİ")
    print("=" * 55)

    print(f"Toplam jet: {results['total_entries']}")
    print(f"UDS Quark: {results['quark_count']}")
    print(f"Gluon: {results['gluon_count']}")
    print(f"Other: {results['other_count']}")

    print("\nÖZELLİK ANALİZLERİ")
    print("-" * 55)

    for feature, values in results["features"].items():

        print(f"\n{feature}")

        print(
            f"Quark mean: "
            f"{values['quark_mean']:.6f}"
        )

        print(
            f"Gluon mean: "
            f"{values['gluon_mean']:.6f}"
        )

        print(
            f"Raw AUC: "
            f"{values['raw_auc']:.6f}"
        )

        print(
            f"Discrimination AUC: "
            f"{values['discrimination_auc']:.6f}"
        )

        print(
            f"KS statistic: "
            f"{values['ks_statistic']:.6f}"
        )

        print(
            f"KS p-value: "
            f"{values['ks_pvalue']:.6e}"
        )

    print("\nAYRIM GÜCÜ SIRALAMASI")
    print("-" * 55)

    for index, feature in enumerate(
        results["ranking"],
        start=1
    ):

        auc = results["features"][feature][
            "discrimination_auc"
        ]

        print(
            f"{index}. {feature}: {auc:.6f}"
        )


# --------------------------------------------------
# 8. ANA PROGRAM
# --------------------------------------------------

def main():

    results = run_reference_analysis()

    print_results(results)

    save_results(results)


if __name__ == "__main__":
    main()
