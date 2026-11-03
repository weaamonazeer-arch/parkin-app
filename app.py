import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Parkinson's Prediction",
    page_icon="🧠"
)


st.title("Parkinson's Disease Prediction")

st.write(
    "أدخلي القيم الصوتية ثم اضغطي على زر التنبؤ."
)

st.warning(
    "هذا التطبيق تعليمي فقط، وليس أداة تشخيص طبي."
)


# تحميل النموذج والـ scaler وأسماء الأعمدة
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_name.pkl")


st.subheader("Enter Patient Features")


# قائمة لتجميع القيم التي تدخلها المستخدم
user_values = []


# إنشاء ثلاثة أعمدة للحقول
screen_columns = st.columns(3)


for i, feature in enumerate(feature_names):

    with screen_columns[i % 3]:

        value = st.number_input(
            feature,
            value=0.0,
            format="%.6f",
            key=feature
        )

        user_values.append(value)


# زر التنبؤ
if st.button("Predict"):

    # تحويل القيم إلى DataFrame بنفس ترتيب التدريب
    input_data = pd.DataFrame(
        [user_values],
        columns=feature_names
    )

    # تطبيق نفس الـ scaling المستخدم أثناء التدريب
    input_data_scaled = scaler.transform(input_data)

    # الحصول على التنبؤ
    prediction = model.predict(input_data_scaled)[0]

    if prediction == 1:
        st.error("The model predicts Parkinson's disease.")
    else:
        st.success("The model predicts that the person is healthy.")
