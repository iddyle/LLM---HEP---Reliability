
import json
from datetime import datetime
from pathlib import Path

import ollama

from analysis_tools import run_reference_analysis


# --------------------------------------------------
# 1. AYARLAR
# --------------------------------------------------

MODEL = "qwen2.5-coder:7b"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "sonuclar"

OUTPUT_JSON = OUTPUT_DIR / "agent_report.json"
OUTPUT_TXT = OUTPUT_DIR / "agent_report.txt"


# --------------------------------------------------
# 2. MODELİN ANALİZ GÖREVİ
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a scientific data analysis assistant
specialized in high-energy particle physics.

You are analyzing a ROOT dataset containing
quark and gluon jet information.

Your task is to interpret the results provided
by Python tools.

Scientific requirements:

1. Distinguish UDS quark jets from gluon jets.
2. Do not include the Other class in the binary comparison.
3. Distinguish raw AUC from direction-independent AUC.
4. Do not describe AUC as classification accuracy.
5. Interpret mean values and distribution differences carefully.
6. Do not claim a physical discovery from these results.
7. Do not invent measurements, variables or observations.
8. Clearly state limitations of the analysis.

Return a JSON object with these fields:

{
    "analysis_summary": "Short summary",
    "feature_interpretations": {
        "QG_ptD": "Physical interpretation",
        "QG_axis2": "Physical interpretation",
        "QG_mult": "Physical interpretation"
    },
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
}

Return only valid JSON.
"""


# --------------------------------------------------
# 3. PYTHON ANALİZ ARACINI ÇALIŞTIR
# --------------------------------------------------

def run_analysis_tool():

    print("\nPython analiz aracı çalıştırılıyor...")

    results = run_reference_analysis()

    print("Python analizi tamamlandı.")

    return results


# --------------------------------------------------
# 4. MODELİN SONUÇLARI YORUMLAMASI
# --------------------------------------------------

def ask_model(analysis_results):

    print("\nQwen analiz sonuçlarını yorumluyor...")

    results_json = json.dumps(
        analysis_results,
        indent=2,
        ensure_ascii=False
    )

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    "Analyze the following Python-generated "
                    "particle physics results.\n\n"
                    + results_json
                )
            }
        ],
        format="json",
        options={
            "temperature": 0
        }
    )

    raw_response = response["message"]["content"]

    return raw_response


# --------------------------------------------------
# 5. MODEL YANITINI JSON OLARAK OKU
# --------------------------------------------------

def parse_report(raw_response):

    try:
        report = json.loads(raw_response)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Model geçerli JSON üretemedi: {error}"
        )

    required_fields = [
        "analysis_summary",
        "feature_interpretations",
        "auc_ranking",
        "statistical_interpretation",
        "limitations",
        "scientific_conclusion"
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in report
    ]

    if missing_fields:
        raise ValueError(
            f"Model yanıtında eksik alanlar: {missing_fields}"
        )

    return report


# --------------------------------------------------
# 6. SONUÇLARI KAYDET
# --------------------------------------------------

def save_results(analysis_results, raw_response, report):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    saved_data = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "condition": "B_tool_assisted",
        "analysis_results": analysis_results,
        "raw_model_response": raw_response,
        "parsed_report": report
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            saved_data,
            file,
            indent=4,
            ensure_ascii=False
        )

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("LLM TOOL-ASSISTED ANALYSIS\n")
        file.write("=" * 60 + "\n\n")

        file.write(f"Model: {MODEL}\n")
        file.write(f"Condition: B - Tool Assisted\n")
        file.write(
            f"Timestamp: {saved_data['timestamp']}\n\n"
        )

        file.write("MODEL REPORT\n")
        file.write("-" * 60 + "\n")

        file.write(
            json.dumps(
                report,
                indent=4,
                ensure_ascii=False
            )
        )

    print("\nSonuçlar kaydedildi:")
    print(OUTPUT_JSON)
    print(OUTPUT_TXT)


# --------------------------------------------------
# 7. ANA PROGRAM
# --------------------------------------------------

def main():

    print("=" * 60)
    print("LLM-BASED PARTICLE PHYSICS ANALYSIS AGENT")
    print("=" * 60)

    print(f"Model: {MODEL}")
    print("Condition: B - Tool Assisted")

    # Python fizik analizini gerçekleştirir
    analysis_results = run_analysis_tool()

    # LLM sonuçları yorumlar
    raw_response = ask_model(analysis_results)

    print("\nModel yanıtı alındı.")

    # Yanıtı kontrol et ve ayrıştır
    report = parse_report(raw_response)

    # Ham ve işlenmiş çıktıları sakla
    save_results(
        analysis_results,
        raw_response,
        report
    )

    print("\n" + "=" * 60)
    print("MODELİN FİZİKSEL ANALİZİ")
    print("=" * 60)

    print(report["analysis_summary"])

    print("\nAUC sıralaması:")

    for index, feature in enumerate(
        report["auc_ranking"],
        start=1
    ):
        print(f"{index}. {feature}")

    print("\nBilimsel sonuç:")
    print(report["scientific_conclusion"])


if __name__ == "__main__":
    main()
