
import json
from datetime import datetime
from pathlib import Path

import ollama

from agent import (
    MODEL,
    run_analysis_tool,
    ask_model,
    parse_report
)


# --------------------------------------------------
# 1. DOSYA AYARLARI
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "sonuclar"

OUTPUT_JSON = OUTPUT_DIR / "deney_c1_self_reflection.json"
OUTPUT_TXT = OUTPUT_DIR / "deney_c1_self_reflection.txt"


# --------------------------------------------------
# 2. SELF-REFLECTION AŞAMASI
# --------------------------------------------------

def self_reflect(analysis_results, initial_report):

    print("\nSelf-Reflection katmanı çalışıyor...")

    analysis_json = json.dumps(
        analysis_results,
        indent=2,
        ensure_ascii=False
    )

    report_json = json.dumps(
        initial_report,
        indent=2,
        ensure_ascii=False
    )

    prompt = f"""
You are reviewing your own previous particle physics analysis.

Your first analysis report was:

{report_json}

The Python-generated analysis data is:

{analysis_json}

Critically review your previous report.

Check specifically:

1. Did you correctly compare quark and gluon means?
2. Did you correctly interpret QG_ptD?
3. Did you correctly interpret QG_axis2?
4. Did you correctly interpret QG_mult?
5. Did you distinguish raw AUC from direction-independent AUC?
6. Did you incorrectly describe AUC as classification accuracy?
7. Did you make any unsupported scientific claims?
8. Are your limitations scientifically appropriate?

Do not assume your previous report was correct.

Use the supplied numerical data to reconsider your interpretations.

Return a corrected report using exactly this JSON structure:

{{
    "analysis_summary": "Short summary",
    "feature_interpretations": {{
        "QG_ptD": "Physical interpretation",
        "QG_axis2": "Physical interpretation",
        "QG_mult": "Physical interpretation"
    }},
    "auc_ranking": [
        "feature 1",
        "feature 2",
        "feature 3"
    ],
    "statistical_interpretation": "Interpretation of AUC and KS",
    "limitations": [
        "limitation 1",
        "limitation 2"
    ],
    "scientific_conclusion": "Overall conclusion"
}}

Return only valid JSON.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are performing a self-review "
                    "of your previous scientific analysis. "
                    "Be critical and do not defend incorrect statements."
                )
            },
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

    return response["message"]["content"]


# --------------------------------------------------
# 3. SONUÇLARI KAYDET
# --------------------------------------------------

def save_results(
    analysis_results,
    initial_report,
    reflected_raw,
    reflected_report
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    data = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "condition": "C1_self_reflection",
        "analysis_results": analysis_results,
        "initial_report": initial_report,
        "reflection_raw_response": reflected_raw,
        "reflected_report": reflected_report
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("C1 - SELF REFLECTION\n")
        file.write("=" * 60 + "\n\n")

        file.write("INITIAL REPORT\n")
        file.write("-" * 60 + "\n")

        file.write(
            json.dumps(
                initial_report,
                indent=4,
                ensure_ascii=False
            )
        )

        file.write("\n\nSELF-REFLECTION REPORT\n")
        file.write("-" * 60 + "\n")

        file.write(
            json.dumps(
                reflected_report,
                indent=4,
                ensure_ascii=False
            )
        )

    print("\nSonuçlar kaydedildi:")
    print(OUTPUT_JSON)
    print(OUTPUT_TXT)


# --------------------------------------------------
# 4. ANA PROGRAM
# --------------------------------------------------

def main():

    print("=" * 60)
    print("C1 - SELF-REFLECTION EXPERIMENT")
    print("=" * 60)

    print(f"Model: {MODEL}")

    # Aynı Python analizini gerçekleştir
    analysis_results = run_analysis_tool()

    # İlk rapor: B koşulundaki başlangıç prompt'u
    print("\nİlk analiz raporu oluşturuluyor...")

    initial_raw = ask_model(analysis_results)
    initial_report = parse_report(initial_raw)

    # Model kendi raporunu yeniden değerlendiriyor
    reflected_raw = self_reflect(
        analysis_results,
        initial_report
    )

    reflected_report = parse_report(reflected_raw)

    # İki raporu kaydet
    save_results(
        analysis_results,
        initial_report,
        reflected_raw,
        reflected_report
    )

    print("\n" + "=" * 60)
    print("SELF-REFLECTION TAMAMLANDI")
    print("=" * 60)

    print("\nİLK RAPORUN AUC SIRALAMASI:")

    for feature in initial_report["auc_ranking"]:
        print(feature)

    print("\nYENİ RAPORUN AUC SIRALAMASI:")

    for feature in reflected_report["auc_ranking"]:
        print(feature)

    print("\nYeni bilimsel sonuç:")
    print(reflected_report["scientific_conclusion"])


if __name__ == "__main__":
    main()
