import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# Load the newly trained model
bundle = joblib.load("student_model.joblib")
model = bundle["model"]
features = bundle["features"]
best_name = bundle["best_name"]
results = bundle["results"]

st.title("🎓 Student Academic Performance Prediction")
st.write(
    "Predict a student's final academic percentage using machine learning."
)

st.info(
    "This prototype uses the UCI Student Performance dataset. "
    "The original data contains Portuguese secondary-school mathematics grades."
)

st.subheader("Enter Student Details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Student Age", min_value=15, max_value=30, value=20
        )

        studytime = st.selectbox(
            "Weekly Study Time",
            [1, 2, 3, 4],
            index=1,
            format_func=lambda x: {
                1: "Less than 2 hours",
                2: "2 to 5 hours",
                3: "5 to 10 hours",
                4: "More than 10 hours"
            }[x]
        )

        failures = st.number_input(
            "Past Class Failures", min_value=0, max_value=4, value=0
        )

        absences = st.number_input(
            "Number of Absences", min_value=0, max_value=100, value=2
        )

    with col2:
        
        sem1_pointer = st.number_input(
            "Semester 1 Pointer",
            min_value=1.0,
            max_value=10.0,
            value=7.0,
            step=0.1,
            format="%.2f"
        )

        sem2_pointer = st.number_input(
            "Semester 2 Pointer",
            min_value=1.0,
            max_value=10.0,
            value=7.0,
            step=0.1,
            format="%.2f"
        )


        activities = st.selectbox(
            "Participates in Extracurricular Activities?", ["yes", "no"]
        )

    submitted = st.form_submit_button(
        "Predict Final Percentage", 
        width = "stretch"
    )

if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "sem1_pointer": sem1_pointer,
        "sem2_pointer": sem2_pointer,
        "activities": activities
    }], columns=features)

    prediction = float(model.predict(input_data)[0])
    prediction = max(0.0, min(100.0, prediction))

    st.subheader("Prediction Result")
    st.metric("Predicted Final Percentage", f"{prediction:.2f}%")
    st.progress(int(round(prediction)))

    if prediction >= 75:
        st.success("High predicted performance")
    elif prediction >= 50:
        st.info("Moderate predicted performance")
    else:
        st.warning("Low predicted performance")

    st.caption(
        "This is a model estimate, not a guaranteed academic result."
    )

st.divider()
st.subheader("Model Performance Comparison")
st.write("Best model selected using the lowest RMSE:", best_name)
st.dataframe(results, use_container_width=True)

st.subheader("RMSE Comparison")
st.bar_chart(results.set_index("Model")[["RMSE (%)"]])

st.caption(
    "The model uses a Portuguese secondary-school dataset, so its predictions "
    "may not generalize to Indian college students."
)
