
import json
import os
from datetime import datetime

import ollama

from agent import MODEL, run_analysis_tool, ask_model, parse_report


# ============================================================
# C3 - CONSTRAINT VALIDATION
# ============================================================

OUTPUT_DIR = "sonuclar"
OUTPUT_JSON = os.path.join(
    OUTPUT_DIR, "deney_c3_constraint_validation.json"
)
OUTPUT_TXT = os.path.join(
    OUTPUT_DIR, "deney_c3_constraint_validation.txt"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# JSON yardımcıları
# ------------------------------------------------------------

def extract_json(text):
    """Model cevabından JSON nesnesini ayıklar."""

    if isinstance(text, dict):
        return text

    text = str(text).strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Model cevabında JSON bulunamadı.")

    return json.loads(text[start:end + 1])


def make_json_safe(value):
    """JSON tarafından desteklenmeyen nesneleri güvenli biçime çevirir."""

    if isinstance(value, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [make_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


# ------------------------------------------------------------
# Model raporunu normalize et
# ------------------------------------------------------------

def normalize_report(response):
    """
    agent.py içindeki parse_report fonksiyonunun
    döndürdüğü raporu standart JSON sözlüğüne çevirir.
    """

    try:
        parsed = parse_report(response)
    except Exception:
        parsed = extract_json(response)

    if isinstance(parsed, str):
        parsed = extract_json(parsed)

    if not isinstance(parsed, dict):
        raise ValueError("Model raporu bir JSON nesnesi değil.")

    return parsed


# ------------------------------------------------------------
# Constraint prompt
# ------------------------------------------------------------

def build_constraint_prompt(results, initial_report):

    return f"""
You are reviewing a particle physics data analysis report.

Your task is to correct the report using the numerical reference
results and the scientific constraints below.

REFERENCE RESULTS:
{json.dumps(results, indent=2, ensure_ascii=False)}

INITIAL REPORT:
{json.dumps(initial_report, indent=2, ensure_ascii=False)}

SCIENTIFIC CONSTRAINTS:

1. QG_ptD:
   - Quark mean = 0.482302
   - Gluon mean = 0.364713
   - Therefore, quark mean is higher.

2. QG_axis2:
   - Quark mean = 0.015294
   - Gluon mean = 0.033201
   - Therefore, gluon mean is higher.

3. QG_mult:
   - Quark mean = 20.952264
   - Gluon mean = 25.703321
   - Therefore, gluon mean is higher.

4. Raw AUC and discrimination AUC are different concepts.
   - QG_ptD raw AUC = 0.231923
   - QG_ptD discrimination AUC = 0.768077
   - Do not confuse these values.

5. AUC is not classification accuracy.

6. A small KS p-value indicates a distribution difference
   under the test assumptions. It does NOT prove a discovery.

7. The analysis compares UDS quark jets with gluon jets.
   Do not generalize the result to every quark flavour.

8. Do not invent measurements, physical properties, or discoveries.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "analysis_summary": "string",
    "feature_interpretations": {{
        "QG_ptD": "string",
        "QG_axis2": "string",
        "QG_mult": "string"
    }},
    "auc_ranking": [
        "QG_axis2",
        "QG_ptD",
        "QG_mult"
    ],
    "statistical_interpretation": "string",
    "limitations": ["string"],
    "scientific_conclusion": "string",

    "mean_direction_claims": {{
        "QG_ptD": "quark_higher",
        "QG_axis2": "gluon_higher",
        "QG_mult": "gluon_higher"
    }},

    "raw_auc_values": {{
        "QG_ptD": 0.231923,
        "QG_axis2": 0.785222,
        "QG_mult": 0.622843
    }},

    "discrimination_auc_values": {{
        "QG_ptD": 0.768077,
        "QG_axis2": 0.785222,
        "QG_mult": 0.622843
    }},

    "auc_is_accuracy": false,
    "ks_proves_discovery": false,
    "analysis_scope": "UDS quark jets compared with gluon jets"
}}
"""


# ------------------------------------------------------------
# Constraint validation
# ------------------------------------------------------------

def validate_constraints(report, results):

    checks = {}

    required_fields = [
        "analysis_summary",
        "feature_interpretations",
        "auc_ranking",
        "statistical_interpretation",
        "limitations",
        "scientific_conclusion",
        "mean_direction_claims",
        "raw_auc_values",
        "discrimination_auc_values",
        "auc_is_accuracy",
        "ks_proves_discovery",
        "analysis_scope"
    ]

    checks["required_fields"] = all(
        field in report for field in required_fields
    )

    if not checks["required_fields"]:
        return {
            "passed": False,
            "checks": checks
        }

    features = results["features"]

    expected_directions = {
        "QG_ptD": "quark_higher",
        "QG_axis2": "gluon_higher",
        "QG_mult": "gluon_higher"
    }

    claims = report["mean_direction_claims"]

    for feature, expected in expected_directions.items():
        checks[f"{feature}_mean_direction"] = (
            claims.get(feature) == expected
        )

    # Raw AUC değerleri
    for feature in features:

        expected = features[feature]["raw_auc"]
        actual = report["raw_auc_values"].get(feature)

        checks[f"{feature}_raw_auc"] = (
            isinstance(actual, (int, float))
            and abs(actual - expected) < 0.001
        )

    # Discrimination AUC değerleri
    for feature in features:

        expected = features[feature]["discrimination_auc"]
        actual = report["discrimination_auc_values"].get(feature)

        checks[f"{feature}_discrimination_auc"] = (
            isinstance(actual, (int, float))
            and abs(actual - expected) < 0.001
        )

    # AUC accuracy değildir.
    checks["auc_not_accuracy"] = (
        report["auc_is_accuracy"] is False
    )

    # KS testi keşif kanıtı değildir.
    checks["ks_not_discovery"] = (
        report["ks_proves_discovery"] is False
    )

    # Analiz kapsamı
    scope = str(report["analysis_scope"]).lower()

    checks["uds_scope"] = (
        "uds" in scope and "gluon" in scope
    )

    # AUC sıralaması
    expected_ranking = [
        "QG_axis2",
        "QG_ptD",
        "QG_mult"
    ]

    checks["auc_ranking"] = (
        report["auc_ranking"] == expected_ranking
    )

    return {
        "passed": all(checks.values()),
        "checks": checks
    }


# ------------------------------------------------------------
# Constraint correction
# ------------------------------------------------------------

def correct_report(results, initial_report):

    prompt = build_constraint_prompt(results, initial_report)

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        options={
            "temperature": 0
        }
    )

    return extract_json(response["message"]["content"])


# ------------------------------------------------------------
# Ana deney
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("C3 - CONSTRAINT VALIDATION")
    print("=" * 60)

    print(f"Model: {MODEL}")

    # 1. Python analizini çalıştır
    print("\nPython analiz aracı çalıştırılıyor...")

    results = run_analysis_tool()

    print("Python analizi tamamlandı.")

    # 2. İlk model raporu
    print("\nİlk model raporu oluşturuluyor...")

    initial_response = ask_model(results)
    initial_report = normalize_report(initial_response)

    print("İlk rapor oluşturuldu.")

    # 3. İlk doğrulama
    print("\nScientific Constraint Validation çalışıyor...")

    validation = validate_constraints(
        initial_report,
        results
    )

    final_report = initial_report
    correction_attempted = False

    # 4. Başarısızsa düzeltme iste
    if not validation["passed"]:

        print("\nBazı kısıtlar sağlanmadı.")
        print("Düzeltme deneniyor...")

        correction_attempted = True

        try:
            corrected_report = correct_report(
                results,
                initial_report
            )

            second_validation = validate_constraints(
                corrected_report,
                results
            )

            final_report = corrected_report
            validation = second_validation

        except Exception as error:

            print("Düzeltme sırasında hata:", error)

    # 5. Sonuçları kaydet
    output = {
        "experiment": "C3 - Constraint Validation",
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "correction_attempted": correction_attempted,
        "initial_report": initial_report,
        "final_report": final_report,
        "validation": validation
    }

    output = make_json_safe(output)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    with open(OUTPUT_TXT, "w", encoding="utf-8") as file:

        file.write("C3 - CONSTRAINT VALIDATION\n")
        file.write("=" * 60 + "\n\n")

        file.write("VALIDATION RESULTS\n")
        file.write("-" * 60 + "\n")

        file.write(
            json.dumps(
                validation,
                indent=4,
                ensure_ascii=False
            )
        )

        file.write("\n\nINITIAL REPORT\n")
        file.write("-" * 60 + "\n")

        file.write(
            json.dumps(
                initial_report,
                indent=4,
                ensure_ascii=False
            )
        )

        file.write("\n\nFINAL REPORT\n")
        file.write("-" * 60 + "\n")

        file.write(
            json.dumps(
                final_report,
                indent=4,
                ensure_ascii=False
            )
        )

    # 6. Terminal sonuçları
    print("\n" + "=" * 60)
    print("C3 KONTROL SONUÇLARI")
    print("=" * 60)

    for name, passed in validation["checks"].items():

        status = "PASS" if passed else "FAIL"

        print(f"{name}: {status}")

    print("\n" + "=" * 60)

    if validation["passed"]:
        print("BÜTÜN KISITLAR SAĞLANDI.")
    else:
        print("BAZI KISITLAR HALA SAĞLANMADI.")

    print("=" * 60)

    print("\nSonuçlar kaydedildi:")
    print(OUTPUT_JSON)
    print(OUTPUT_TXT)


if __name__ == "__main__":
    main()
