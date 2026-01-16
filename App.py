import joblib
import pandas as pd
import gradio as gr
import os

# Model dosyası yolu
MODEL_PATH = "models/cardiorisk_rf_tuned.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model bulunamadı: {MODEL_PATH}\n"
        "Önce main.py dosyasını çalıştırıp modeli eğitmen gerekiyor."
    )

model = joblib.load(MODEL_PATH)

# main.py'deki ile aynı feature engineering fonksiyonu
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bins = [0, 50, 60, 70, 120]
    labels = ['<50', '50-60', '60-70', '>70']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    df['hyponatremia_flag'] = (df['serum_sodium'] < 135).astype(int)
    return df


def predict_risk(
    age,
    anaemia,
    creatinine_phosphokinase,
    diabetes,
    ejection_fraction,
    high_blood_pressure,
    platelets,
    serum_creatinine,
    serum_sodium,
    sex,
    smoking,
    time
):
    # Kullanıcı inputlarını DataFrame'e çevir
    data = {
        "age": [age],
        "anaemia": [int(anaemia)],
        "creatinine_phosphokinase": [creatinine_phosphokinase],
        "diabetes": [int(diabetes)],
        "ejection_fraction": [ejection_fraction],
        "high_blood_pressure": [int(high_blood_pressure)],
        "platelets": [platelets],
        "serum_creatinine": [serum_creatinine],
        "serum_sodium": [serum_sodium],
        "sex": [int(sex)],
        "smoking": [int(smoking)],
        "time": [time],
    }

    df = pd.DataFrame(data)
    df = feature_engineering(df)

    # Tahmin
    prob = model.predict_proba(df)[0][1]  # ölüm olayı olasılığı (1 sınıfı)
    pred = int(prob >= 0.5)

    risk_percent = round(prob * 100, 2)

    if risk_percent < 30:
        risk_level = "Low"
    elif risk_percent < 60:
        risk_level = "Medium"
    else:
        risk_level = "High"

    text = (
        f"Predicted death event: **{pred}**\n\n"
        f"Risk probability: **{risk_percent}%**\n"
        f"Risk level: **{risk_level}**"
    )

    return text


# Gradio arayüzü
def build_interface():
    description = """
    # 🫀 CardioRisk - Heart Failure Risk Prediction

    Bu arayüz kalp yetmezliği hastaları için ölüm riskini tahmin eden bir ML modelini kullanır.
    Model: RandomForest (tuned) + scikit-learn pipeline  
    Dataset: Heart Failure Clinical Records

    > Not: Bu bir **hackathon / demo** projesidir, tıbbi tavsiye yerine geçmez.
    """

    inputs = [
        gr.Slider(20, 100, value=60, step=1, label="Age"),
        gr.Radio(["0", "1"], value="0", label="Anaemia (0=No, 1=Yes)"),
        gr.Slider(50, 8000, value=200, step=10, label="Creatinine Phosphokinase"),
        gr.Radio(["0", "1"], value="0", label="Diabetes (0=No, 1=Yes)"),
        gr.Slider(10, 80, value=30, step=1, label="Ejection Fraction"),
        gr.Radio(["0", "1"], value="0", label="High Blood Pressure (0=No, 1=Yes)"),
        gr.Slider(50000, 800000, value=250000, step=1000, label="Platelets"),
        gr.Slider(0.5, 10.0, value=1.2, step=0.1, label="Serum Creatinine"),
        gr.Slider(110, 150, value=135, step=1, label="Serum Sodium"),
        gr.Radio(["0", "1"], value="1", label="Sex (0=Female, 1=Male)"),
        gr.Radio(["0", "1"], value="0", label="Smoking (0=No, 1=Yes)"),
        gr.Slider(0, 300, value=120, step=1, label="Follow-up Time (days)"),
    ]

    output = gr.Markdown()

    iface = gr.Interface(
        fn=predict_risk,
        inputs=inputs,
        outputs=output,
        title="CardioRisk - Heart Failure Risk Prediction",
        description=description,
    )
    return iface


if __name__ == "__main__":
    iface = build_interface()
    iface.launch()
