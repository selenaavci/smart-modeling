# 🤖 Smart Modeling Agent

## ⚡ Hızlı Başlangıç (Local)

Proje Windows ve Linux/macOS üzerinde doğrudan localden çalışacak şekilde hazırlanmıştır. Tek gereksinim **Python 3.10+**.

### 🐧 Linux / 🍎 macOS
```bash
./run.sh
```

### 🪟 Windows
```bat
run.bat
```

Her iki script de ilk çalıştırmada:
1. `.venv` adında bir sanal ortam oluşturur
2. `requirements.txt` içindeki bağımlılıkları yükler
3. Streamlit uygulamasını (`app.py`) başlatır

Tarayıcınız otomatik olarak `http://localhost:8501` adresinde açılacaktır.

### 📦 Dosya Yükleme Limiti
`.streamlit/config.toml` dosyasında `maxUploadSize = 5120` (MB) olarak ayarlanmıştır — yani **5 GB**'a kadar CSV / XLSX / XLS / XML dosyası yükleyebilirsiniz.

### 🗂️ Klasör Yapısı
```
smart-modeling/
├── app.py                    # Streamlit ana arayüz (sihirbaz)
├── requirements.txt          # Python bağımlılıkları
├── run.sh / run.bat          # Linux-macOS / Windows başlatıcılar
├── .streamlit/config.toml    # 5 GB upload + tema ayarları
└── src/
    ├── data_loader.py        # CSV / Excel / XML okuma
    ├── problem_detector.py   # Problem tipi önerisi
    ├── preprocessing.py      # Otomatik temizlik + encoding
    ├── model_engine.py       # Model eğitim & karşılaştırma
    ├── visualizations.py     # Grafikler
    ├── exporter.py           # Excel çıktı
    └── model_info.py         # Kullanıcı dostu model açıklamaları
```

---

## 🚀 Projenin Amacı

Smart Modeling Agent, kullanıcıların kendi veri setleri üzerinde makine öğrenmesi modellerini teknik bilgiye ihtiyaç duymadan çalıştırabilmesini sağlayan bir self-service AI aracıdır.

Agent; veri setini analiz ederek uygun problem tipini (classification, regression, clustering) önerir, farklı algoritmaları otomatik olarak çalıştırır, performanslarını karşılaştırır ve en uygun modeli kullanıcıya sunar. Aynı zamanda görselleştirme ve detaylı çıktı seçenekleri ile karar destek sürecini güçlendirir.

---

## 🎯 Temel Kullanım Senaryoları

- Müşteri churn tahmini  
- Satış tahmini  
- Fraud detection (classification)  
- Müşteri segmentasyonu (clustering)  
- Operasyonel tahminleme ve risk analizi  
- Hızlı model denemeleri (baseline ML)  

---

## 🧠 Agent Yaklaşımı

Agent, kullanıcıyı adım adım yönlendiren bir **Guided ML Workflow** mantığıyla çalışır.

---

## 🔄 End-to-End Workflow

1. Dataset Upload (CSV / Excel)
2. Veri analizi ve problem tipi önerisi
3. Kullanıcının problem tipini seçmesi (override edilebilir)
4. Target (hedef) kolon seçimi
5. Feature seçimleri (opsiyonel)
6. Model seçimi (opsiyonel)
7. Model training ve evaluation
8. Model karşılaştırma ve en iyi modelin belirlenmesi
9. Görselleştirme ve sonuç analizi
10. Excel export

---

## 🧩 Problem Tipi Tespiti

Agent veri setini analiz ederek öneride bulunur:

- Target kolon kategorik → **Classification önerilir**
- Target kolon sayısal → **Regression önerilir**
- Target kolon yok → **Clustering önerilir**

Kullanıcı isterse bu öneriyi değiştirebilir.

---

## ⚙️ Desteklenen Model Türleri

### 🔹 Classification

- Logistic Regression  
- Random Forest Classifier  
- Gradient Boosting (XGBoost / LightGBM opsiyonel)  

**Kullanılan metrikler:**
- Accuracy  
- F1 Score (öncelikli)  
- Precision / Recall  
- ROC-AUC  

---

### 🔹 Regression

- Linear Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  

**Kullanılan metrikler:**
- RMSE  
- MAE  
- R² Score  

---

### 🔹 Clustering

- K-Means  
- DBSCAN  

**Kullanılan metrikler:**
- Silhouette Score  
- Inertia (K-Means için)  

---

## 📊 Model Karşılaştırma

Tüm modeller aşağıdaki gibi karşılaştırılır:

| Model | Metric | Score |
|------|-------|------|

- En iyi model otomatik olarak highlight edilir  
- Kullanıcı farklı modellerin performansını karşılaştırabilir  

---

## 📈 Görselleştirme (Visualization Layer)

Agent, model sonuçlarını görsel olarak sunar:

### Classification:
- Confusion Matrix  
- ROC Curve  
- Feature Importance  

### Regression:
- Actual vs Predicted Plot  
- Residual Plot  
- Feature Importance  

### Clustering:
- Cluster dağılım grafikleri  
- PCA / 2D projection  

Kullanıcı, farklı modeller arasında geçiş yaparak görselleri karşılaştırabilir.

---

## 🔍 Feature Importance

- Tree-based modeller için otomatik hesaplanır  
- En etkili feature’lar görsel olarak sunulur  
- Model yorumlanabilirliğini artırır  

---

## ⚠️ Overfitting Kontrolü

Agent, model performansını analiz eder:

- Train vs Test skorları karşılaştırılır  
- Büyük fark varsa kullanıcı uyarılır  

---

## 🧹 Otomatik Veri Ön İşleme

Agent aşağıdaki işlemleri otomatik gerçekleştirir:

- Eksik değer handling  
- Kategorik değişken encoding (One-Hot Encoding)  
- Feature scaling (gerekli durumlarda)  
- Veri tipi dönüşümleri  

---

## 📥 Excel Export

Kullanıcı tüm sonuçları Excel olarak indirebilir:

- Model karşılaştırma tablosu  
- Prediction sonuçları  
- Evaluation metrikleri  
- Seçilen best model çıktısı  

---

## 🏗️ Teknik Mimari

### Veri Girişi
- CSV / XLSX upload  
- Dosya validasyonu  

### Ön İşleme Katmanı
- Data cleaning  
- Encoding  
- Scaling  

### Model Engine
- Model training  
- Evaluation  
- Comparison  

### Visualization Layer
- Grafik üretimi  
- Model bazlı dashboard  

### Output Layer
- Excel export  
- Sonuç özetleri  

---

## 🧪 Kullanılan Teknolojiler

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost / LightGBM (opsiyonel)  
- Matplotlib / Seaborn  
- Streamlit  

---

## 🔮 Faz 2 Geliştirmeleri

- SHAP ile gelişmiş model açıklamaları  
- Hyperparameter tuning (GridSearch / Optuna)  
- Model kayıt ve versiyonlama (MLflow)  
- LLM ile model yorumlama  
- Otomatik feature engineering önerileri  

---

## ⚠️ Riskler ve Dikkat Edilmesi Gerekenler

- Yanlış target seçimi model performansını düşürür  
- Küçük veri setlerinde model overfitting yapabilir  
- Imbalanced datasetlerde accuracy yanıltıcı olabilir  
- Clustering sonuçları domain bilgisi olmadan zor yorumlanabilir  

---

## 💡 Business Impact

- Teknik bilgiye ihtiyaç duymadan ML model geliştirme imkanı sağlar  
- Veri odaklı karar alma süreçlerini hızlandırır  
- Farklı model alternatiflerini hızlıca karşılaştırma imkanı sunar  
- Veri bilimi ekiplerine olan bağımlılığı azaltır  

---

## 🧠 CTO Takeaway

Smart Modeling Agent, kurum içinde makine öğrenmesi yetkinliğini demokratize ederek business kullanıcıların kendi verileri üzerinde model geliştirmesini mümkün kılar. Model karşılaştırma, görselleştirme ve yorumlanabilirlik özellikleri sayesinde yalnızca analiz değil, aksiyon alınabilir içgörüler üretir.

---

## 📌 Not

Bu agent şu anda Streamlit tabanlı demo olarak geliştirilmiştir ve henüz production ortamına alınmamıştır. Gelecek aşamada merkezi sunucuya deploy edilerek kurum genelinde erişilebilir hale getirilmesi planlanmaktadır.
