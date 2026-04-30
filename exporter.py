import json
import io
import pandas as pd


def export_to_excel(original_df, labels, profiles, llm_result, selected_features):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        result_df = original_df.copy()
        result_df["Segment"] = labels
        result_df.to_excel(writer, sheet_name="Veri ve Segmentler", index=False)

        profile_rows = []
        for cid, p in profiles.items():
            row = {"Segment": cid, "Kayit Sayisi": p["size"], "Yuzde": p["percentage"]}
            for feat, stats in p["stats"].items():
                if "mean" in stats:
                    row[f"{feat}_ortalama"] = stats["mean"]
                    row[f"{feat}_genel_ort"] = stats["overall_mean"]
                    row[f"{feat}_fark_%"] = stats["diff_from_overall_pct"]
                elif "dominant_value" in stats:
                    row[f"{feat}_baskin_deger"] = stats["dominant_value"]
                    row[f"{feat}_baskin_%"] = stats["dominant_pct"]
            profile_rows.append(row)

        pd.DataFrame(profile_rows).to_excel(writer, sheet_name="Segment Profilleri", index=False)

        if llm_result and "segments" in llm_result:
            llm_rows = []
            for seg in llm_result["segments"]:
                llm_rows.append({
                    "Segment ID": seg.get("id", ""),
                    "Segment Adi": seg.get("name", ""),
                    "Profil": seg.get("profile", ""),
                    "Davranis Analizi": seg.get("behavioral_analysis", ""),
                    "Icguruler": " | ".join(seg.get("key_insights", [])),
                    "Aksiyonlar": " | ".join(seg.get("recommended_actions", [])),
                    "Riskler": " | ".join(seg.get("risk_notes", [])),
                })
            pd.DataFrame(llm_rows).to_excel(writer, sheet_name="LLM Yorumlari", index=False)

            summary_df = pd.DataFrame([{
                "Yonetici Ozeti": llm_result.get("executive_summary", ""),
                "Segmentler Arasi Icguruler": " | ".join(llm_result.get("cross_segment_insights", [])),
            }])
            summary_df.to_excel(writer, sheet_name="Yonetici Ozeti", index=False)

    output.seek(0)
    return output


def export_to_json(profiles, llm_result):
    export_data = {
        "profiles": {},
        "llm_interpretation": llm_result,
    }

    for cid, p in profiles.items():
        export_data["profiles"][str(cid)] = {
            "size": p["size"],
            "percentage": p["percentage"],
            "stats": p["stats"],
            "distinguishing_features": p["distinguishing_features"],
        }

    return json.dumps(export_data, ensure_ascii=False, indent=2)
