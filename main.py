import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def load_and_preprocess_data():
    # Read the CSV file
    df = pd.read_csv('Predict Student Dropout and Academic Success (1).csv', sep=';', encoding='utf-8-sig')
    
    # Define feature columns in a specific order that will be maintained
    feature_columns = [
        'Age at enrollment',
        'Scholarship holder',
        'Previous qualification (grade)',
        'Admission grade',
        'Curricular units 1st sem (grade)',
        'Curricular units 1st sem (approved)',
        'Curricular units 2nd sem (grade)',
        'Curricular units 2nd sem (approved)',
        'Tuition fees up to date',
        'Gender'
    ]
    
    # Create feature and target datasets
    X = df[df['Target'] != 'Enrolled'][feature_columns].copy()
    y = df[df['Target'] != 'Enrolled']['Target']
    
    # Handle categorical variables
    le = LabelEncoder()
    X['Gender'] = le.fit_transform(X['Gender'])
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    return df, X, y, feature_columns

def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def create_prediction_inputs(feature_columns):
    prediction_inputs = {}
    
    # Create input fields in the same order as feature_columns
    for feature in feature_columns:
        if feature == 'Age at enrollment':
            prediction_inputs[feature] = st.sidebar.slider('Usia saat Mendaftar', 17, 70, 20)
        elif feature == 'Scholarship holder':
            prediction_inputs[feature] = st.sidebar.selectbox('Penerima Beasiswa', [0, 1])
        elif feature == 'Previous qualification (grade)':
            prediction_inputs[feature] = st.sidebar.slider('Nilai Kualifikasi Sebelumnya', 0.0, 200.0, 120.0)
        elif feature == 'Admission grade':
            prediction_inputs[feature] = st.sidebar.slider('Nilai Masuk', 0.0, 200.0, 120.0)
        elif feature == 'Curricular units 1st sem (grade)':
            prediction_inputs[feature] = st.sidebar.slider('Nilai Semester 1', 0.0, 20.0, 12.0)
        elif feature == 'Curricular units 1st sem (approved)':
            prediction_inputs[feature] = st.sidebar.slider('Mata Kuliah Lulus Semester 1', 0, 10, 5)
        elif feature == 'Curricular units 2nd sem (grade)':
            prediction_inputs[feature] = st.sidebar.slider('Nilai Semester 2', 0.0, 20.0, 12.0)
        elif feature == 'Curricular units 2nd sem (approved)':
            prediction_inputs[feature] = st.sidebar.slider('Mata Kuliah Lulus Semester 2', 0, 10, 5)
        elif feature == 'Tuition fees up to date':
            prediction_inputs[feature] = st.sidebar.selectbox('Status Pembayaran UKT', [0, 1])
        elif feature == 'Gender':
            prediction_inputs[feature] = st.sidebar.selectbox('Jenis Kelamin', [0, 1])
    
    return prediction_inputs

def main():
    
    st.set_page_config(page_title="Analisis & Prediksi Performa Akademik Mahasiswa")
    
    st.title("Dashboard Analisis & Prediksi Performa Akademik Mahasiswa")
    
    try:
        # Load and prepare data
        df, X, y, feature_columns = load_and_preprocess_data()
        
        # Display the dataset at the top of the page
        st.header("Dataset Mahasiswa")
        st.dataframe(df)  # Display the first 10 rows of the dataset
        
        model, scaler = train_model(X, y)
        
        # Sidebar for predictions
        st.sidebar.header("Prediksi Status Mahasiswa")
        
        # Create input fields in the correct order
        prediction_inputs = create_prediction_inputs(feature_columns)
        
        if st.sidebar.button('Prediksi Status'):
            # Create DataFrame with features in correct order
            input_data = pd.DataFrame([prediction_inputs])[feature_columns]
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            # Map probabilities to classes
            classes = model.classes_  # Get the order of classes from the model
            proba_dict = {classes[i]: prediction_proba[0][i] for i in range(len(classes))}
            
            # Display prediction results
            st.sidebar.subheader("Hasil Prediksi")
            st.sidebar.write(f"Status Prediksi: {prediction[0]}")
            st.sidebar.write(f"Probabilitas Graduate: {proba_dict.get('Graduate', 0):.2%}")
            st.sidebar.write(f"Probabilitas Dropout: {proba_dict.get('Dropout', 0):.2%}")
        
        # Main dashboard content
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribusi Status Mahasiswa")
            fig_target = px.pie(df, names='Target', title='Distribusi Status Mahasiswa')
            st.plotly_chart(fig_target)
            
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig_importance = px.bar(importance_df, x='Importance', y='Feature', 
                                  orientation='h', title='Pentingnya Fitur dalam Prediksi')
            st.plotly_chart(fig_importance)
        
        with col2:
            st.subheader("Performa Akademik vs Status")
            fig_grades = px.scatter(df, 
                                  x='Curricular units 1st sem (grade)',
                                  y='Curricular units 2nd sem (grade)',
                                  color='Target',
                                  title='Perbandingan Nilai Semester 1 vs Semester 2')
            st.plotly_chart(fig_grades)
        
        # Display model performance metrics
        st.header("Metrik Performa Model")
        col3, col4 = st.columns(2)
        
        with col3:
            st.metric("Akurasi Model", f"{model.score(X, y):.2%}")
        
        with col4:
            st.metric("Jumlah Fitur yang Digunakan", len(feature_columns))
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        st.info("Pastikan file data berada di lokasi yang benar dan dalam format yang sesuai.")

if __name__ == "__main__":
    main()
