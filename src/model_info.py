"""Model, problem tipi ve metrik açıklamaları — teknik ama tarafsız dil."""
from __future__ import annotations


MODEL_DESCRIPTIONS = {
    # Classification
    "Logistic Regression": (
        "Doğrusal bir sınıflandırma modelidir. Hızlı çalışır ve yorumlanması "
        "kolaydır. Karmaşık olmayan ilişkilerde iyi sonuç verir; ilk deneme "
        "için uygun bir başlangıç modelidir."
    ),
    "Random Forest (Classifier)": (
        "Çok sayıda karar ağacından oluşan bir topluluk modelidir. Tahminler "
        "ağaçların çoğunluk oyuyla belirlenir. Karmaşık verilerde stabil "
        "sonuçlar üretir ve overfitting'e karşı dirençlidir."
    ),
    "Gradient Boosting (Classifier)": (
        "Sıralı şekilde eğitilen küçük karar ağaçlarından oluşur. Her yeni "
        "ağaç, bir öncekinin hatalarını azaltmaya çalışır. Genellikle yüksek "
        "doğruluk sağlar, eğitim süresi daha uzun olabilir."
    ),

    # Regression
    "Linear Regression": (
        "Girdi ve çıktı arasında doğrusal bir ilişki olduğunu varsayan bir "
        "regresyon modelidir. Yorumu kolaydır; basit tahmin problemleri için "
        "uygun bir başlangıç modelidir."
    ),
    "Random Forest (Regressor)": (
        "Çok sayıda karar ağacından gelen tahminlerin ortalamasını alır. "
        "Sayısal tahminlerde sağlam sonuçlar verir ve aykırı değerlere karşı "
        "dirençlidir."
    ),
    "Gradient Boosting (Regressor)": (
        "Sıralı ağaçlarla tahmin yapar; her adımda önceki hatayı azaltmaya "
        "odaklanır. Karmaşık sayısal ilişkilerde güçlü performans gösterir."
    ),

    # Clustering
    "K-Means": (
        "Verileri birbirine yakın noktalardan oluşan K adet kümeye ayırır. "
        "Küme sayısının önceden belirlenmesi gerekir. Segmentasyon "
        "senaryoları için yaygın kullanılan bir algoritmadır."
    ),
    "DBSCAN": (
        "Yoğunluk tabanlı bir kümeleme algoritmasıdır. Sık komşusu olan "
        "noktalar aynı kümeye atanır; izole noktalar gürültü olarak "
        "işaretlenir. Küme sayısının önceden bilinmesi gerekmez. `eps` "
        "parametresi sonuçları belirgin şekilde etkiler."
    ),
}


PROBLEM_DESCRIPTIONS = {
    "classification": (
        "**Sınıflandırma (Classification)**\n\n"
        "Bir kaydın hangi kategoriye ait olduğunu tahmin eder. Örnek "
        "senaryolar: müşteri ayrılır mı / kalır mı, işlem dolandırıcılık mı "
        "değil mi. Hedef sütunun evet/hayır veya A/B/C gibi sınırlı sayıda "
        "etiketten oluşması beklenir."
    ),
    "regression": (
        "**Regresyon (Regression)**\n\n"
        "Bir sayısal değeri tahmin eder. Örnek senaryolar: satış tahmini, "
        "fiyat tahmini, süre tahmini. Hedef sütunun sürekli bir sayısal "
        "değişken olması beklenir."
    ),
    "clustering": (
        "**Kümeleme (Clustering)**\n\n"
        "Verileri birbirine benzer gruplara ayırır. Tahmin edilecek bir "
        "hedef sütun yoksa kullanılır. Örnek senaryolar: müşteri "
        "segmentasyonu, benzer kayıtların gruplanması."
    ),
}


METRIC_DESCRIPTIONS = {
    "Accuracy": "Doğru tahminlerin toplam tahminlere oranı. 1'e yakın olması iyidir.",
    "F1 (weighted)": "Precision ve Recall'ın dengeli birleşimi. Sınıf dağılımı dengesiz olduğunda Accuracy'den daha güvenilir bir metriktir.",
    "Precision": "Pozitif olarak tahmin edilen kayıtların ne kadarının gerçekten pozitif olduğunu gösterir. Yanlış alarm oranıyla ters orantılıdır.",
    "Recall": "Gerçek pozitif kayıtların ne kadarının modelce yakalanabildiğini gösterir. Kaçırılan pozitiflerin önemli olduğu durumlarda öncelikli metriktir.",
    "ROC-AUC": "Modelin pozitif ve negatif sınıfları ayırt edebilme kapasitesini ölçer. 1 mükemmel, 0.5 rastgele tahmine eşdeğerdir.",
    "Train Accuracy": "Eğitim kümesi üzerindeki doğruluk. Test doğruluğundan belirgin şekilde yüksekse overfitting işareti olabilir.",
    "RMSE": "Tahmin hatasının karekök ortalaması. Küçük olması iyidir ve hedef değişkenle aynı birimdedir.",
    "MAE": "Ortalama mutlak hata. Küçük olması iyidir; aykırı değerlere RMSE'den daha az duyarlıdır.",
    "R2": "Modelin hedefteki değişkenliğin ne kadarını açıkladığını gösterir. 1'e yakın iyi, 0 ortalama tahmine eşdeğer, negatif değerler ortalamadan daha kötü demektir.",
    "Train R2": "Eğitim kümesi üzerindeki R². Test R²'sinden belirgin şekilde yüksekse overfitting işareti olabilir.",
    "Silhouette": "Kümelerin ne kadar iyi ayrıldığını ölçer. -1 ile 1 arasındadır; 1'e yakın değerler iyi ayrışmış kümeleri gösterir.",
    "Inertia": "K-Means için iç toplam kare hatası. Küçük olması daha sıkı kümeleri gösterir, ancak küme sayısı arttıkça sürekli düşer.",
    "Küme Sayısı": "Model tarafından oluşturulan küme adedi.",
    "Gürültü Noktası": "DBSCAN tarafından hiçbir kümeye dahil edilmeyen (gürültü olarak işaretlenen) kayıt sayısı.",
}
