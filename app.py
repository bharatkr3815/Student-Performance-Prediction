import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# Load Model
model = joblib.load("models/student_performance_model.pkl")

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.title {
    font-size: 36px;
    font-weight: bold;
    color: #2563eb;
}
.subtitle {
    color: #64748b;
    font-size: 17px;
}
.result-card {
    background-color: #e0f2fe;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
}
.result-text {
    font-size: 28px;
    font-weight: bold;
    color: #0369a1;
}
</style>
""", unsafe_allow_html=True)


# Sidebar
st.sidebar.title("🎓 Student Predictor")
st.sidebar.write("Machine Learning Project")
st.sidebar.divider()

st.sidebar.subheader("About Project")
st.sidebar.write(
    "This application predicts student final marks "
    "using Machine Learning."
)

st.sidebar.divider()

st.sidebar.subheader("Model Information")
st.sidebar.write("Algorithm: Linear Regression")
st.sidebar.write("Feature Selection: SelectKBest")
st.sidebar.write("Evaluation: 5-Fold Cross Validation")

st.sidebar.info(
    "This model provides estimated marks, "
    "not guaranteed results."
)


# Main Header
st.markdown(
    '<p class="title">Student Performance Prediction</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict student academic performance using Machine Learning.</p>',
    unsafe_allow_html=True
)

st.divider()

# Model Performance Dashboard

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="R² Score",
        value="0.0678"
    )

with col2:
    st.metric(
        label="Mean Absolute Error",
        value="3.2936"
    )

with col3:
    st.metric(
        label="Mean Squared Error",
        value="18.7322"
    )

st.caption(
    "Performance metrics calculated using 5-Fold Cross Validation."
)

st.divider()

# Prediction History Initialization

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

    
# Student Input Form
st.subheader("📝 Enter Student Details")

with st.form("student_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Student Age",
            min_value=15,
            max_value=22,
            value=16
        )

        studytime = st.selectbox(
            "Weekly Study Time",
            [1, 2, 3, 4]
        )

    with col2:
        failures = st.number_input(
            "Past Class Failures",
            min_value=0,
            max_value=4,
            value=0
        )

        absences = st.number_input(
            "Number of Absences",
            min_value=0,
            max_value=75,
            value=0
        )

    submit = st.form_submit_button(
        "🎯 Predict Final Marks",
        use_container_width=True
    )


# Prediction
if submit:

    input_data = pd.DataFrame({
        'age': [age],
        'studytime': [studytime],
        'failures': [failures],
        'absences': [absences]
    })

    prediction = model.predict(input_data)[0]

    # Store Prediction History

    st.session_state.prediction_history.append({
        "Age": age,
        "Study Time": studytime,
        "Failures": failures,
        "Absences": absences,
        "Predicted Marks": round(prediction, 2)
    })

    st.divider()
    st.subheader("📊 Prediction Result")

    st.markdown(
        f"""
        <div class="result-card">
            <p class="result-text">
                🎯 {prediction:.2f} / 20
            </p>
            <p>Predicted Final Marks</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Progress Bar
    progress = max(0, min(float(prediction), 20)) / 20

    st.progress(progress)

    st.caption(
        "This is an estimated prediction, not a guaranteed result."
    )

    # Prediction History Display

    st.divider()
    
    st.subheader("📜 Prediction History")
    
    if len(st.session_state.prediction_history) > 0:
    
        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )
    
        st.dataframe(
            history_df,
            use_container_width=True
        )

                # Convert History into CSV
        csv = history_df.to_csv(index=False).encode('utf-8')

        # Download Button
        st.download_button(
            label="📥 Download Prediction History",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear Prediction History"):
            st.session_state.prediction_history = []
            st.rerun()
    
    else:
    
        st.info("No prediction history available yet.")

 
# Footer
st.divider()

st.caption(
    "Student Performance Prediction | Built with Python, "
    "Scikit-learn and Streamlit"
)