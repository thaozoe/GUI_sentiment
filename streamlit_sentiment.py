import numpy as np
import pandas as pd
import os
import pickle
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

# Load models
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_vietnamese(text):
    # Chuyển thành chữ thường
    text = text.lower()
    # Loại bỏ ký tự đặc biệt
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text

# Tải mô hình
models = {
    "Gradient Boosting Machine (GBM)": load_model('GBM_model.pkl'),
    "Random Forest": load_model('Randomforest_model.pkl'),
    "Support Vector Machine (SVM)": load_model('SVM_model.pkl')
}
vectorizer = load_model('count_vectorizer.pkl')  # Sử dụng CountVectorizer.pkl
model_prediction = load_model('randomforest_model_np.pkl')

# Đọc file dữ liệu
data = pd.read_csv("Danh_gia_with_label.csv")

# Hàm đọc file txt
def display_file_content(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except Exception as e:
            return f"Không thể đọc file. Lỗi: {e}"
    else:
        return f"File không tồn tại: {file_path}"

# Hàm dự đoán sentiment
def predict_sentiment(input_text, model_prediction, vectorizer, threshold=0.4):
    # Tiền xử lý và vector hóa
    input_text = preprocess_vietnamese(input_text)
    transformed_text = vectorizer.transform([input_text])
    # Dự đoán xác suất
    probabilities = model_prediction.predict_proba(transformed_text)
    # Tinh chỉnh ngưỡng
    if probabilities[0][1] > threshold:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'
    return sentiment


#--------------
# GUI
st.title("Sentiment Analysis for Hasaki")
st.write("## Positive vs Negative Review")
st.image("hasaki_banner.jpg")
menu = ["Business Objective", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
Nguyễn Thị Mỷ Tiên  
Đặng Thị Thảo  
""", unsafe_allow_html=True)
st.sidebar.write("""#### Giảng viên hướng dẫn: Cô Khuất Thùy Phương """)
st.sidebar.write("""#### Ngày báo cáo thực hiện: 16/12/2024""")
if choice == 'Business Objective': 
    st.subheader("Business Objective")
    file_path = ("Hasaki_objective.txt")
    content = display_file_content(file_path)
    st.write(content)

    st.write("#### Thuật toán sử dụng: Machine Learning truyền thống")
    

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Some data")
    st.dataframe(data[['ma_san_pham', 'noi_dung_binh_luan','sentiment_label']].head(5))
    st.dataframe(data[['ma_san_pham', 'noi_dung_binh_luan','sentiment_label']].tail(5))  
    st.write("##### 2. Visualize positive and negative sentiment")
    st.write("###### *Tỉ lệ đánh giá positive và negative* ")
    st.image("sentiment_percentage_pie_chart.png")  
    st.write("###### *Các chủ đề bình luận trên Hasaki* ") 
    st.image("sentiment_distribution_chart.png")
    st.write("###### *Các chủ đề bình luận trên Hasaki tương ứng với phản ứng của khách hàng* ")
    st.image("sentiment_topics_chart.png")
    st.write("###### *Analysis sentiment by year* ")
    st.image("sentiment_by_year.png")
    st.write("###### *Analysis sentiment by month* ")
    st.image("sentiment_by_month.png")
    st.write("###### *Analysis sentiment by hour* ")
    st.image("sentiment_by_hour.png")
    st.write("###### *Wordcloud of positive and negative comment* ")
    st.image("wordcloud_pos.png")
    st.image("wordcloud_neg.png")
    st.write("##### 3. Dự đoán với Mô hình Machine Learning")
    st.write("Chọn mô hình để xem kết quả dự đoán:")
    model_name = st.selectbox("Chọn mô hình:", list(models.keys()))
    selected_model = models[model_name]
    st.write(f"Bạn đã chọn: **{model_name}**")
    if model_name == 'Gradient Boosting Machine (GBM)':
        file_path = ("GBM_model_result.txt")
        content = display_file_content(file_path)
        st.code(content)
        st.write('Confusion Matrix- GBM:')
        st.image('confusion_matrix_for_GBM.png')
    elif model_name == "Support Vector Machine (SVM)":
        file_path = ("SVM_model _result.txt")
        content = display_file_content(file_path)
        st.code(content)
        st.write('Confusion Matrix- SVM:')
        st.image('confusion_matrix_for_SVM.png')
    else:
        file_path = ("Random_Forest_result.txt")
        content = display_file_content(file_path)
        st.code(content)
        st.write('Confusion Matrix- Random Forest:')
        st.image('confusion_matrix_for_RF.png')
    st.write("##### 5. Summary:")
    file_path = ("model_evaluation.txt")
    content = display_file_content(file_path)
    st.code(content)


elif choice == 'New Prediction':
    st.subheader("Chọn dữ liệu và dự đoán sentiment")

    # Chọn phương thức nhập dữ liệu
    data_input_type = st.radio("Bạn muốn nhập dữ liệu như thế nào?", options=("Input", "Upload"))

    if data_input_type == "Input":
        # Nhập bình luận từ người dùng
        if "ten_san_pham" not in data.columns:
            st.error("Dữ liệu không có cột 'ten_san_pham'. Vui lòng kiểm tra lại file dữ liệu!")
        else:
            # Chọn sản phẩm
            product_name = st.selectbox(
                "Chọn sản phẩm:",
                options=data['ten_san_pham'].unique()
            )
        new_comment = st.text_area("Nhập bình luận mới:")

        if new_comment:
            # Dự đoán sentiment
            new_comment = preprocess_vietnamese(new_comment)
            sentiment = predict_sentiment(new_comment, model_prediction, vectorizer)
            st.success(f"Dự đoán sentiment: **{sentiment}**")

    elif data_input_type == "Upload":
        # Tải lên file
        uploaded_file = st.file_uploader("Chọn file (txt hoặc csv):", type=['txt', 'csv'])

        if uploaded_file:
            try:
                # Đọc dữ liệu từ file
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file, header=None, names=['text'])
                elif uploaded_file.name.endswith('.txt'):
                    data = pd.read_csv(uploaded_file, header=None, names=['text'], sep="\n")

                # Hiển thị dữ liệu đầu vào
                st.subheader("Dữ liệu tải lên:")
                st.dataframe(data.head())

                # Xử lý giá trị NaN và đảm bảo dữ liệu là chuỗi
                data['text'] = data['text'].fillna('').astype(str)
                # Chuẩn bị dữ liệu và dự đoán
                texts = data['text'].tolist()
                processed_texts = [preprocess_vietnamese(text) for text in texts]
                transformed_texts = vectorizer.transform(processed_texts)
                predictions = model_prediction.predict(transformed_texts)

                # Kết quả dự đoán
                data['Prediction'] = predictions
                data['Prediction'] = data['Prediction'].apply(lambda x: 'Positive' if x == 1 else 'Negative')

                st.subheader("Kết quả dự đoán:")
                st.dataframe(data)

                # Tải xuống file kết quả
                st.download_button(
                    label="Tải xuống kết quả",
                    data=data.to_csv(index=False).encode('utf-8'),
                    file_name='predictions.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Lỗi khi xử lý file: {e}")
        else:
            st.warning("Vui lòng tải lên một file hợp lệ!")

    else:
        st.warning("Vui lòng chọn phương thức nhập dữ liệu!")
