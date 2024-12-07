import numpy as np
import pandas as pd
import os
import pickle
import streamlit as st
from underthesea import word_tokenize
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

# Load models
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Tải mô hình
models = {
    # "Gradient Boosting Machine (GBM)": load_model('GBM_model.pkl'),
    "Random Forest": load_model('Randomforest_model.pkl'),
    "Support Vector Machine (SVM)": load_model('SVM_model.pkl')
}
vectorizer_ = load_model('vectorizer.pkl')

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

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download('stopwords')

# Khởi tạo Lemmatizer và stopwords từ NLTK (Tiếng Việt có thể phải tải riêng)
lemmatizer = WordNetLemmatizer()

# Nếu stopwords tiếng Việt không có trong NLTK, bạn có thể tải từ một file khác hoặc sử dụng tiếng Anh
try:
    stop_words = set(stopwords.words('vietnamese'))  # Dùng stopwords tiếng Việt nếu có
except:
    stop_words = set(stopwords.words('english'))  # Dùng stopwords tiếng Anh nếu không có tiếng Việt

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    # Loại bỏ ký tự đặc biệt (dấu câu)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tách từ bằng thư viện underthesea
    text = word_tokenize(text)
    
    # Loại bỏ stopwords (từ dừng)
    text = [word for word in text if word not in stop_words]
    
    # Lemmatization (chuyển từ về dạng gốc)
    text = [lemmatizer.lemmatize(word) for word in text]
    
    # Kết hợp lại thành chuỗi
    return ' '.join(text)

# Hàm tải mô hình và vectorizer từ file pickle
def load_model_and_vectorizer():
    try:
        # Tải mô hình Random Forest
        with open('Randomforest_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        # Tải vectorizer
        with open('vectorizer.pkl', 'rb') as vec_file:
            vectorizer = pickle.load(vec_file)
        
        return model, vectorizer
    except FileNotFoundError as e:
        st.error("Không tìm thấy tệp mô hình hoặc vectorizer.")
        st.exception(e)
        st.stop()  # Dừng chương trình nếu không tìm thấy file
    except Exception as e:
        st.error("Đã xảy ra lỗi trong khi tải mô hình hoặc vectorizer.")
        st.exception(e)
        st.stop()  # Dừng chương trình nếu có lỗi khác

# Hàm dự đoán sentiment
def predict_sentiment(comment, model, vectorizer):
    try:
        # Tiền xử lý bình luận mới
        processed_comment = preprocess_text(comment)
        
        # Biến đổi bình luận đã tiền xử lý thành vector
        comment_transformed = vectorizer.transform([processed_comment])
        
        # Dự đoán sentiment
        prediction = model.predict(comment_transformed)
        
        # Hiển thị kết quả
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        return sentiment
    except Exception as e:
        st.error("Đã xảy ra lỗi trong quá trình dự đoán.")
        st.exception(e)
        return None


#--------------
# GUI
st.title("Sentiment Analysis for Hasaki")
st.write("## Positive vs Negative Review")
st.image("hasaki_banner.jpg")
menu = ["Business Objective", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
Nguyễn Thị Mỷ Tiên  
<br><br>
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
    # Bước 1: Kiểm tra và chọn tên sản phẩm
    if "ten_san_pham" not in data.columns:
        st.error("Dữ liệu không có cột 'ten_san_pham'. Vui lòng kiểm tra lại file dữ liệu!")
    else:
        product_name = st.selectbox(
            "Chọn sản phẩm:",
            options=data['ten_san_pham'].unique()
        )

        # Nhập bình luận mới từ người dùng
        new_comment = st.text_area("Nhập bình luận mới:")

        # Kiểm tra nếu người dùng đã nhập bình luận
        if new_comment:
            # Tải mô hình và vectorizer
            model, vectorizer = load_model_and_vectorizer()

            # Dự đoán sentiment
            sentiment = predict_sentiment(new_comment, model, vectorizer)
            
            if sentiment:
                st.write(f"**Dự đoán sentiment cho bình luận:** {sentiment}")
