# Smart Modeling Agent

## Hizli Baslangic (Local)

Proje Windows ve Linux/macOS uzerinde dogrudan localden calisacak sekilde hazirlanmistir. Tek gereksinim **Python 3.10+**.

### Linux / macOS
```bash
./run.sh
```

### Windows
```bat
run.bat
```

Her iki script de ilk calistirmada:
1. `.venv` adinda bir sanal ortam olusturur
2. `requirements.txt` icindeki bagimliliklari yukler
3. Streamlit uygulamasini (`app.py`) baslatir

Tarayiciniz otomatik olarak `http://localhost:8501` adresinde acilacaktir.

> **Not:** `sma-mod/` klasoru eski Streamlit surumlerine (1.26.0) uyumlu, Windows RDP ortamlari icin optimize edilmis versiyondur. Headless mod, ASCII-safe Turkce karakterler ve `use_container_width` API'si kullanir.

### Dosya Yukleme Limiti
`.streamlit/config.toml` dosyasinda `maxUploadSize = 5120` (MB) olarak ayarlanmistir — yani **5 GB**'a kadar CSV / XLSX / XLS / XML dosyasi yukleyebilirsiniz.

### Klasor Yapisi
```
smart-modeling/
├── app.py                    # Streamlit ana arayuz (sihirbaz)
├── requirements.txt          # Python bagimliliklari
├── run.sh / run.bat          # Linux-macOS / Windows baslaticilar
├── .streamlit/config.toml    # 5 GB upload + tema ayarlari
├── ornek_veri/               # Ornek veri setleri (churn, ev fiyatlari, musteri segmentasyonu)
├── sma-mod/                  # Windows RDP uyumlu versiyon (Streamlit 1.26.0)
└── src/
    ├── data_loader.py        # CSV / Excel / XML okuma
    ├── problem_detector.py   # Problem tipi onerisi (classification / regression)
    ├── preprocessing.py      # Otomatik temizlik + encoding + scaling
    ├── model_engine.py       # Model egitim & karsilastirma
    ├── visualizations.py     # Grafikler (confusion matrix, ROC, residual, feature importance)
    ├── exporter.py           # Excel cikti
    └── model_info.py         # Kullanici dostu model aciklamalari
```

---

## Projenin Amaci

Smart Modeling Agent, kullanicilarin kendi veri setleri uzerinde makine ogrenmesi modellerini teknik bilgiye ihtiyac duymadan calistirabilmesini saglayan bir self-service AI aracidir.

Agent; veri setini analiz ederek uygun problem tipini (classification veya regression) onerir, farkli algoritmalari otomatik olarak calistirir, performanslarini karsilastirir ve en uygun modeli kullaniciya sunar. Ayni zamanda gorsellistirme ve detayli cikti secenekleri ile karar destek surecini guclendirir.

---

## Temel Kullanim Senaryolari

- Musteri churn tahmini
- Satis tahmini
- Fraud detection (classification)
- Operasyonel tahminleme ve risk analizi
- Hizli model denemeleri (baseline ML)

---

## Agent Yaklasimi

Agent, kullaniciyi adim adim yonlendiren bir **Guided ML Workflow** mantigiyla calisir.

---

## End-to-End Workflow

1. Dataset Upload (CSV / Excel / XML)
2. Veri analizi ve onizleme
3. Veri profili (numerik istatistikler, eksik degerler, kategorik kardinalite)
4. Problem tipi onerisi (classification veya regression)
5. Kullanicinin problem tipini secmesi (override edilebilir)
6. Tanimlayici (ID) kolon secimi
7. Hedef (target) kolon secimi
8. Feature secimleri (opsiyonel)
9. Egitim secenekleri (cross-validation, standardizasyon)
10. Model secimi (opsiyonel)
11. Model training ve evaluation
12. Model karsilastirma ve en iyi modelin belirlenmesi
13. Gorsellistirme ve sonuc analizi
14. Modeling artifact (ZIP) indirme
15. Excel export

---

## Problem Tipi Tespiti

Agent veri setini analiz ederek oneride bulunur:

- Target kolon numerik ve benzersiz deger sayisi > 20 ve benzersiz oran > %2 → **Regression onerilir**
- Diger durumlarda → **Classification onerilir**

Kullanici isterse bu oneriyi degistirebilir.

---

## Desteklenen Model Turleri

### Classification

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier (sklearn)

**Kullanilan metrikler:**
- Accuracy
- F1 Score (weighted) — **en iyi model secim kriteri**
- Precision (weighted)
- Recall (weighted)
- ROC-AUC (yalnizca ikili siniflandirma)
- Train Accuracy
- CV Accuracy mean/std (cross-validation aciksa)

---

### Regression

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor (sklearn)

**Kullanilan metrikler:**
- RMSE
- MAE
- R2 Score — **en iyi model secim kriteri**
- Train R2
- CV R2 mean/std (cross-validation aciksa)

---

## Model Karsilastirma

Tum modeller tablo halinde karsilastirilir:

| Model | Metric | Score |
|------|-------|------|

- En iyi model otomatik olarak highlight edilir
- Kullanici farkli modellerin performansini karsilastirabilir

---

## Gorsellistirme (Visualization Layer)

Agent, model sonuclarini gorsel olarak sunar:

### Classification:
- Confusion Matrix
- ROC Curve (ikili siniflandirma icin)
- Feature Importance

### Regression:
- Actual vs Predicted Plot
- Residual Plot
- Feature Importance

Kullanici, farkli modeller arasinda gecis yaparak gorselleri karsilastirabilir.

---

## Feature Importance

- Tree-based modeller icin `feature_importances_` kullanilir
- Linear modeller icin `coef_` mutlak degeri kullanilir
- En etkili ilk 15 feature gorsel olarak sunulur

---

## Overfitting Kontrolu

Agent, model performansini analiz eder:

- Train vs Test skorlari karsilastirilir
- Test skoru > 0.97 ise uyari verilir
- Train-Test fark > 0.15 ise uyari verilir

---

## Otomatik Veri On Isleme

Agent asagidaki islemleri otomatik gerceklestirir:

- Eksik deger handling (numerik: median, kategorik: mode)
- Kategorik degisken encoding (One-Hot Encoding — `pd.get_dummies`)
- Feature scaling (StandardScaler, kullanici actiginda)
- Datetime kolonlari string'e cevirme
- Stratified train-test split (uygun durumlarda)

---

## Modeling Artifact (ZIP)

Kullanici egitim sonuclarini ZIP paketi olarak indirebilir:

- `metadata.json` — Calistirma ozeti ve parametreler
- `features.json` — Secilen ve engineered feature listesi
- `model.pkl` — Egitilmis en iyi model (pickle)
- `X_train.csv` / `X_test.csv` — Egitim ve test setleri
- `y_train.csv` / `y_test.csv` — Hedef degerleri
- `README.txt` — Paket aciklamasi

> Model dosyasi (.pkl) dogrudan canli sisteme yuklenmemelidir; once MLOps pipeline'inda ek validasyondan gecmelidir.

---

## Excel Export

Kullanici tum sonuclari Excel olarak indirebilir:

- Model karsilastirma tablosu
- Train-test ayirimi (satir bazinda)
- Tum veri seti uzerinde tahminler (train + test birlesik)

---

## Desteklenen Dosya Formatlari

- CSV (otomatik ayirici ve encoding tespiti: virgul, noktali virgul, tab; UTF-8, Latin-1)
- XLSX / XLS (Excel)
- XML (tabular yapi gerektir)

---

## Teknik Mimari

### Veri Girisi
- CSV / XLSX / XLS / XML upload
- Dosya validasyonu

### On Isleme Katmani
- Data cleaning (missing value imputation)
- One-Hot Encoding
- Scaling (opsiyonel)

### Model Engine
- Model training
- Evaluation
- Cross-validation (opsiyonel, 5-fold)
- Comparison

### Visualization Layer
- Grafik uretimi (matplotlib + seaborn, dark tema)
- Model bazli dashboard

### Output Layer
- Modeling artifact ZIP
- Excel export
- Sonuc ozetleri

---

## Kullanilan Teknolojiler

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit
- openpyxl / xlsxwriter (Excel export)
- lxml (XML okuma)

---

## Faz 2 Gelistirmeleri

- Clustering destegi (K-Means, DBSCAN)
- XGBoost / LightGBM entegrasyonu
- SHAP ile gelismis model aciklamalari
- Hyperparameter tuning (GridSearch / Optuna)
- Model kayit ve versiyonlama (MLflow)
- LLM ile model yorumlama
- Otomatik feature engineering onerileri

---

## Riskler ve Dikkat Edilmesi Gerekenler

- Yanlis target secimi model performansini dusurur
- Kucuk veri setlerinde model overfitting yapabilir
- Imbalanced datasetlerde accuracy yaniltici olabilir
- Bu uygulama bir hizli prototipleme / kesif aracidir — egitilen modeller dogrudan production'a alinmak icin uygun degildir

---

## Business Impact

- Teknik bilgiye ihtiyac duymadan ML model gelistirme imkani saglar
- Veri odakli karar alma sureclerini hizlandirir
- Farkli model alternatiflerini hizlica karsilastirma imkani sunar
- Veri bilimi ekiplerine olan bagimliligi azaltir

---

## Not

Bu agent su anda Streamlit tabanli demo olarak gelistirilmistir ve henuz production ortamina alinmamistir. Gelecek asamada merkezi sunucuya deploy edilerek kurum genelinde erisilebilir hale getirilmesi planlanmaktadir.
