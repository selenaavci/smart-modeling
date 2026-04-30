"""
Smart Modeling için örnek veri seti oluşturucu.

Her problem tipi (classification, regression, clustering) için gerçekçi,
modellerin öğrenebilmesi için yeterli sinyal içeren veri setleri üretir.

Çalıştırma:
    python ornek_veri/olustur.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent


# ---------------------------------------------------------------------------
# 1) Classification — Müşteri Churn
# ---------------------------------------------------------------------------
def siniflandirma_musteri_churn(seed: int = 42, n: int = 2000) -> pd.DataFrame:
    """İkili sınıflandırma: müşterinin hizmeti bırakıp bırakmayacağı."""
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 80, n)
    tenure = rng.integers(1, 73, n)
    monthly = np.round(rng.uniform(20, 120, n), 2)
    total = np.round((monthly * tenure + rng.normal(0, 50, n)).clip(min=0), 2)
    contract = rng.choice(
        ["Aylik", "1 Yillik", "2 Yillik"], n, p=[0.55, 0.25, 0.20]
    )
    internet = rng.choice(["DSL", "Fiber", "Yok"], n, p=[0.35, 0.45, 0.20])
    payment = rng.choice(
        ["Kredi Karti", "Havale", "Elektronik Cek", "Otomatik Odeme"], n
    )
    has_dependents = rng.choice(["Evet", "Hayir"], n, p=[0.30, 0.70])

    logit = (
        -2.0
        + 1.6 * (tenure < 12).astype(float)
        + 1.2 * (contract == "Aylik").astype(float)
        + 0.9 * (monthly > 80).astype(float)
        + 0.6 * (internet == "Fiber").astype(float)
        + 0.3 * (age > 55).astype(float)
        - 0.8 * (has_dependents == "Evet").astype(float)
        + rng.normal(0, 0.4, n)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.uniform(0, 1, n) < prob).astype(int)

    return pd.DataFrame(
        {
            "musteri_id": np.arange(1, n + 1),
            "yas": age,
            "musteri_suresi_ay": tenure,
            "aylik_ucret": monthly,
            "toplam_ucret": total,
            "sozlesme_tipi": contract,
            "internet_servisi": internet,
            "odeme_yontemi": payment,
            "bakmakla_yukumlu": has_dependents,
            "churn": churn,
        }
    )


# ---------------------------------------------------------------------------
# 1b) Classification — Çalışan Churn (İK / Attrition)
# ---------------------------------------------------------------------------
def siniflandirma_calisan_churn(seed: int = 45, n: int = 1800) -> pd.DataFrame:
    """İkili sınıflandırma: çalışanın işten ayrılıp ayrılmayacağı (attrition)."""
    rng = np.random.default_rng(seed)

    yas = rng.integers(22, 60, n)
    departman = rng.choice(
        ["Yazilim", "Pazarlama", "Finans", "Operasyon", "Satis", "IK"],
        n,
        p=[0.28, 0.15, 0.13, 0.17, 0.20, 0.07],
    )
    pozisyon = rng.choice(
        ["Junior", "Uzman", "Kidemli", "Yonetici"],
        n,
        p=[0.30, 0.40, 0.22, 0.08],
    )
    kidem_yili = rng.integers(0, 20, n)

    pozisyon_carpani = np.where(
        pozisyon == "Junior", 1.0,
        np.where(pozisyon == "Uzman", 1.4,
        np.where(pozisyon == "Kidemli", 1.9, 2.6)),
    )
    aylik_maas = np.round(
        (25_000 * pozisyon_carpani + kidem_yili * 1_500 + rng.normal(0, 3_000, n)).clip(min=22_000),
        -2,
    ).astype(int)

    performans_puani = np.round(rng.normal(75, 12, n)).clip(40, 100).astype(int)
    devamsizlik_gun = rng.integers(0, 25, n)
    fazla_mesai = rng.choice(["Evet", "Hayir"], n, p=[0.30, 0.70])
    son_terfi_yili = np.minimum(kidem_yili, rng.integers(0, 8, n))
    egitim_saat = rng.integers(0, 60, n)
    is_tatmini = rng.integers(1, 6, n)  # 1–5
    calisma_modeli = rng.choice(["Ofis", "Hibrit", "Uzaktan"], n, p=[0.45, 0.40, 0.15])
    sehir = rng.choice(
        ["Istanbul", "Ankara", "Izmir", "Bursa", "Antalya"],
        n,
        p=[0.55, 0.20, 0.12, 0.08, 0.05],
    )

    logit = (
        -2.2
        + 1.4 * (fazla_mesai == "Evet").astype(float)
        + 1.5 * (is_tatmini <= 2).astype(float)
        - 0.7 * (is_tatmini >= 4).astype(float)
        + 1.2 * (kidem_yili < 2).astype(float)
        + 0.9 * (son_terfi_yili >= 4).astype(float)
        + 0.6 * (yas < 28).astype(float)
        + 0.5 * (devamsizlik_gun > 12).astype(float)
        + 0.4 * ((performans_puani >= 90) & (son_terfi_yili >= 3)).astype(float)
        - 0.4 * (egitim_saat > 30).astype(float)
        - 0.3 * (calisma_modeli == "Hibrit").astype(float)
        + rng.normal(0, 0.4, n)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    ayrildi = (rng.uniform(0, 1, n) < prob).astype(int)

    return pd.DataFrame(
        {
            "calisan_id": np.arange(1, n + 1),
            "yas": yas,
            "departman": departman,
            "pozisyon": pozisyon,
            "kidem_yili": kidem_yili,
            "aylik_maas": aylik_maas,
            "performans_puani": performans_puani,
            "devamsizlik_gun": devamsizlik_gun,
            "fazla_mesai": fazla_mesai,
            "son_terfi_yili": son_terfi_yili,
            "egitim_saat": egitim_saat,
            "is_tatmini": is_tatmini,
            "calisma_modeli": calisma_modeli,
            "sehir": sehir,
            "ayrildi": ayrildi,
        }
    )


# ---------------------------------------------------------------------------
# 2) Regression — Ev Fiyatları
# ---------------------------------------------------------------------------
def regresyon_ev_fiyatlari(seed: int = 43, n: int = 1500) -> pd.DataFrame:
    """Sayısal tahmin: ev özelliklerinden fiyat tahmini."""
    rng = np.random.default_rng(seed)

    area = rng.integers(40, 350, n)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.integers(1, 4, n)
    building_age = rng.integers(0, 50, n)
    neighborhood = rng.choice(
        ["Kadikoy", "Besiktas", "Uskudar", "Sisli", "Atasehir", "Kartal"], n
    )
    balcony = rng.choice(["Var", "Yok"], n, p=[0.75, 0.25])
    parking = rng.choice(["Var", "Yok"], n, p=[0.60, 0.40])

    premium_map = {
        "Besiktas": 1.45,
        "Kadikoy": 1.30,
        "Sisli": 1.35,
        "Uskudar": 1.20,
        "Atasehir": 1.15,
        "Kartal": 1.00,
    }
    premium = np.array([premium_map[x] for x in neighborhood])

    price = (
        area * 55_000
        + bedrooms * 120_000
        + bathrooms * 90_000
        - building_age * 12_000
        + (balcony == "Var").astype(float) * 75_000
        + (parking == "Var").astype(float) * 130_000
    ) * premium
    price = price + rng.normal(0, 200_000, n)
    price = np.round(price.clip(min=500_000), -3).astype(int)

    return pd.DataFrame(
        {
            "ev_id": np.arange(1, n + 1),
            "alan_m2": area,
            "oda_sayisi": bedrooms,
            "banyo_sayisi": bathrooms,
            "bina_yasi": building_age,
            "semt": neighborhood,
            "balkon": balcony,
            "otopark": parking,
            "fiyat_tl": price,
        }
    )


# ---------------------------------------------------------------------------
# 3) Clustering — Müşteri Segmentasyonu
# ---------------------------------------------------------------------------
def kumeleme_musteri_segmentasyonu(seed: int = 44, n_per_cluster: int = 300) -> pd.DataFrame:
    """Denetimsiz: 4 belirgin müşteri segmenti."""
    rng = np.random.default_rng(seed)

    #            [gelir,    harcama, yas, aylik_islem]
    centers = np.array(
        [
            [25_000, 25, 24, 5],     # Genç, düşük gelir, düşük harcama
            [90_000, 85, 35, 25],    # Orta yaş, yüksek gelir, yüksek harcama
            [60_000, 45, 45, 12],    # Orta yaş, orta gelir, orta harcama
            [120_000, 30, 55, 8],    # Üst yaş, yüksek gelir, tasarruflu
        ],
        dtype=float,
    )
    stds = np.array(
        [
            [5_000, 8, 4, 2],
            [12_000, 7, 6, 5],
            [8_000, 6, 5, 3],
            [15_000, 6, 6, 2],
        ],
        dtype=float,
    )

    blocks = []
    for center, std in zip(centers, stds):
        block = rng.normal(center, std, size=(n_per_cluster, 4))
        blocks.append(block)
    X = np.vstack(blocks)

    df = pd.DataFrame(
        X,
        columns=["yillik_gelir_tl", "harcama_skoru", "yas", "aylik_islem_sayisi"],
    )
    df["yillik_gelir_tl"] = df["yillik_gelir_tl"].round(-2).clip(lower=10_000).astype(int)
    df["harcama_skoru"] = df["harcama_skoru"].round(1).clip(0, 100)
    df["yas"] = df["yas"].round().clip(18, 80).astype(int)
    df["aylik_islem_sayisi"] = df["aylik_islem_sayisi"].round().clip(lower=0).astype(int)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "musteri_id", np.arange(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    datasets = {
        "siniflandirma_churn.csv": siniflandirma_musteri_churn(),
        "siniflandirma_calisan_churn.csv": siniflandirma_calisan_churn(),
        "regresyon_ev_fiyatlari.csv": regresyon_ev_fiyatlari(),
        "kumeleme_musteri_segmentasyonu.csv": kumeleme_musteri_segmentasyonu(),
    }

    for name, df in datasets.items():
        path = OUT / name
        df.to_csv(path, index=False)
        print(f"[CSV] {name}: {len(df):,} satir x {len(df.columns)} sutun -> {path}")

    # XML örneği: XML yükleme akışını test etmek için küçük bir alt küme
    xml_df = siniflandirma_musteri_churn().head(500)
    xml_path = OUT / "siniflandirma_churn.xml"
    xml_df.to_xml(xml_path, index=False, root_name="musteriler", row_name="musteri")
    print(f"[XML] siniflandirma_churn.xml: {len(xml_df):,} satir -> {xml_path}")


if __name__ == "__main__":
    main()
