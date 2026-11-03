
import json
from datetime import datetime
from pathlib import Path

from agent import MODEL, run_analysis_tool, ask_model, parse_report

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "sonuclar"

REFERENCE_FILE = OUTPUT_DIR / "referans_analiz.json"
OUTPUT_JSON = OUTPUT_DIR / "deney_c2_execution_verification.json"
OUTPUT_TXT = OUTPUT_DIR / "deney_c2_execution_verification.txt"


def verify_execution(results):
    """Python analiz çıktısının bütünlüğünü ve sayısal tutarlılığını kontrol eder."""

    checks = {}

    required_keys = [
        "dataset",
        "tree",
        "total_entries",
        "quark_count",
        "gluon_count",
        "other_count",
        "features",
        "ranking"
    ]

    checks["required_fields"] = all(
        key in results for key in required_keys
    )

    checks["class_counts"] = (
        results["total_entries"]
        == results["quark_count"]
        + results["gluon_count"]
        + results["other_count"]
    )

    checks["feature_completeness"] = all(
        feature in results["features"]
        for feature in ["QG_ptD", "QG_axis2", "QG_mult"]
    )

    checks["numeric_validity"] = True

    for feature, values in results["features"].items():
        for key in [
            "quark_mean",
            "gluon_mean",
            "raw_auc",
            "discrimination_auc",
            "ks_statistic",
            "ks_pvalue"
        ]:
            value = values.get(key)

            if not isinstance(value, (int, float)):
                checks["numeric_validity"] = False
                continue

            if not float("-inf") < value < float("inf"):
                checks["numeric_validity"] = False

    checks["auc_consistency"] = all(
        abs(
            values["discrimination_auc"]
            - max(values["raw_auc"], 1 - values["raw_auc"])
        ) < 1e-8
        for values in results["features"].values()
    )

    checks["auc_range"] = all(
        0 <= values["raw_auc"] <= 1
        and 0 <= values["discrimination_auc"] <= 1
        for values in results["features"].values()
    )

    checks["ranking_consistency"] = (
        results["ranking"]
        == sorted(
            results["features"],
            key=lambda feature: results["features"][feature][
                "discrimination_auc"
            ],
            reverse=True
        )
    )

    checks["mean_direction_consistency"] = all(
        (
            values["mean_direction"] == "quark_higher"
            and values["quark_mean"] > values["gluon_mean"]
        )
        or (
            values["mean_direction"] == "gluon_higher"
            and values["gluon_mean"] > values["quark_mean"]
        )
        for values in results["features"].values()
    )

    passed = all(checks.values())

    return {
        "passed": passed,
        "checks": checks
    }


def compare_with_reference(results, reference):
    """Yeni çalıştırmayı kayıtlı referans analizle karşılaştırır."""

    comparisons = {}

    comparisons["dataset_match"] = (
        results["dataset"] == reference["dataset"]
    )

    comparisons["entry_count_match"] = (
        results["total_entries"] == reference["total_entries"]
    )

    comparisons["class_counts_match"] = (
        results["quark_count"] == reference["quark_count"]
        and results["gluon_count"] == reference["gluon_count"]
        and results["other_count"] == reference["other_count"]
    )

    comparisons["feature_values_match"] = True

    for feature in reference["features"]:
        for key in [
            "quark_mean",
            "gluon_mean",
            "raw_auc",
            "discrimination_auc",
            "ks_statistic"
        ]:
            current = results["features"][feature][key]
            expected = reference["features"][feature][key]

            if abs(current - expected) > 1e-8:
                comparisons["feature_values_match"] = False

    return comparisons


def main():
    print("=" * 60)
    print("C2 - EXECUTION VERIFICATION")
    print("=" * 60)

    print(f"Model: {MODEL}")

    with open(REFERENCE_FILE, "r", encoding="utf-8") as file:
        reference = json.load(file)

    # 1. Python analizini çalıştır
    results = run_analysis_tool()

    # 2. Çalıştırma ve veri bütünlüğü kontrolleri
    print("\nExecution Verification çalışıyor...")

    verification = verify_execution(results)
    reference_comparison = compare_with_reference(results, reference)

    print("\nKONTROL SONUÇLARI")
    print("-" * 60)

    for name, passed in verification["checks"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print("\nREFERANS KARŞILAŞTIRMASI")
    print("-" * 60)

    for name, passed in reference_comparison.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    all_passed = (
        verification["passed"]
        and all(reference_comparison.values())
    )

    if not all_passed:
        raise RuntimeError(
            "Execution Verification başarısız. "
            "Model yorumlama aşamasına geçilmedi."
        )

    # 3. Doğrulanmış sonuçları modele gönder
    print("\nBütün kontroller başarılı.")
    print("Qwen doğrulanmış sonuçları yorumluyor...")

    raw_report = ask_model(results)
    report = parse_report(raw_report)

    # 4. Sonuçları kaydet
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "condition": "C2_execution_verification",
        "verification": verification,
        "reference_comparison": reference_comparison,
        "analysis_results": results,
        "model_report": report
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as file:
        file.write("C2 - EXECUTION VERIFICATION\n")
        file.write("=" * 60 + "\n\n")

        file.write("VERIFICATION RESULTS\n")
        file.write(json.dumps(
            verification, indent=4, ensure_ascii=False
        ))

        file.write("\n\nREFERENCE COMPARISON\n")
        file.write(json.dumps(
            reference_comparison, indent=4, ensure_ascii=False
        ))

        file.write("\n\nMODEL REPORT\n")
        file.write(json.dumps(
            report, indent=4, ensure_ascii=False
        ))

    print("\nSonuçlar kaydedildi:")
    print(OUTPUT_JSON)
    print(OUTPUT_TXT)

    print("\nC2 EXECUTION VERIFICATION TAMAMLANDI.")


if __name__ == "__main__":
    main()
