import json

import requests
import streamlit as st


SYSTEM_PROMPT = """Sen bir veri analisti ve iş zekâsı uzmanısın.
Clustering (segmentasyon) sonuçlarını yorumlaman isteniyor.

Görevlerin:
1. Her segmente anlamlı ve akılda kalıcı bir isim ver
2. Her segment için davranışsal profil açıklaması yaz
3. Segmentler arasındaki farkları vurgula
4. Her segment için iş birimi için anlamlı içgörüler üret
5. Her segment için aksiyon önerileri sun
6. Risk veya dikkat noktalarını belirt

Çıktını aşağıdaki JSON formatında ver. Türkçe yaz.
{
  "segments": [
    {
      "id": 0,
      "name": "Segment Adı",
      "profile": "Profil açıklaması...",
      "behavioral_analysis": "Davranış analizi...",
      "key_insights": ["içgörü 1", "içgörü 2"],
      "recommended_actions": ["aksiyon 1", "aksiyon 2"],
      "risk_notes": ["risk 1"]
    }
  ],
  "executive_summary": "Yönetici özeti...",
  "cross_segment_insights": ["segmentler arası içgörü 1", "içgörü 2"]
}"""


def _secret(key: str) -> str:
    try:
        return st.secrets.get(key, "") if hasattr(st, "secrets") else ""
    except Exception:
        return ""


def interpret_segments(cluster_summary, context=""):
    base_url = _secret("LLM_BASE_URL").strip().rstrip("/")
    model = _secret("LLM_MODEL").strip()
    api_key = _secret("LLM_API_KEY").strip()

    if not base_url or not model:
        return {
            "raw_response": (
                "⚠️ **LLM bilgileri eksik.**\n\n"
                "Yapay zekâ yorumunu etkinleştirmek için `LLM_API_KEY`, "
                "`LLM_BASE_URL` ve `LLM_MODEL` alanlarını Streamlit Cloud → "
                "**Settings → Secrets** bölümüne ya da yerelde "
                "`.streamlit/secrets.toml` dosyasına ekleyin.\n\n"
                f"Kümeleme özeti LLM'e gönderilmeye hazır ({len(cluster_summary)} karakter)."
            )
        }

    user_message = f"Aşağıdaki clustering sonuçlarını yorumla:\n\n{cluster_summary}"
    if context:
        user_message += f"\n\nEk bağlamsal bilgi: {context}"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 4096,
        "temperature": 0.7,
    }

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=120,
    )
    response.raise_for_status()
    response_text = response.json()["choices"][0]["message"]["content"]

    try:
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return {"raw_response": response_text}
