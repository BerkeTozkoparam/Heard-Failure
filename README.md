<img width="1452" height="817" alt="Ekran Resmi 2026-01-16 23 21 01" src="https://github.com/user-attachments/assets/561cce68-fb49-446c-a721-bdf03e87f5ee" />
# 🫀 CardioRisk — Heart Failure Risk Prediction

CardioRisk, kalp yetmezliği hastalarında **ölüm riskini tahmin eden** makine öğrenimi tabanlı demo bir projedir.  
Model, klinik parametreleri giriş olarak alır ve hastanın **DEATH_EVENT** olasılığını üretir.

> ⚠️ Bu proje bir **hackathon / akademik demo** çalışmasıdır. Tıbbi karar destek sistemi olarak kullanılmaz.

---

## 📌 **Dataset**

Kullanılan dataset: **Heart Failure Clinical Records**  
Kaynak: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records

**Özellikler (excerpt):**

| Feature | Açıklama |
|---|---|
| age | Yaş |
| anaemia | Kansızlık (0/1) |
| creatinine_phosphokinase | CPK değeri |
| diabetes | Diyabet (0/1) |
| ejection_fraction | Kalp EF değeri (%) |
| serum_creatinine | Kreatinin |
| serum_sodium | Sodyum |
| time | Takip süresi (gün) |
| DEATH_EVENT | Hedef (1=Ölüm) |

---

## 🧠 **Model**

Model pipeline içeriği:

- Feature Engineering
    - `age_group`
    - `hyponatremia_flag`
- Preprocessing
    - StandardScaler (numeric)
    - OneHotEncoder (categorical)
- Classifier
    - `RandomForestClassifier` (Hyperparameter Tuned)

---

## 🎯 **Accuracy**

Test Accuracy: 0.7333
ROC-AUC Skoru: 0.7548


Skorlar veri bölünmesine göre değişebilir.

---

## 🏗️ **Proje Yapısı**

```bash
Heard-Failure/
├── Main.py                # Model training + tuning
├── app.py                 # Gradio serving UI
├── requirements.txt       # Dependencies
├── heart_failure...csv    # Dataset
└── models/
    └── cardiorisk_rf.joblib

##🚀 Kurulum
git clone https://github.com/BerkeTozkoparam/Heard-Failure.git
cd Heard-Failure
pip install -r requirements.txt




