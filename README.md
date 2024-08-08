TÊN ĐỀ TÀI: DIFFERENTIAL PRIVACY IN DEEP LEARNING
---------------------------------------------------------------------
Sinh viên thực hiện:
- Lê Minh Trí - 20133100
---------------------------------------------------------------------
Mục tiêu: Nghiên cứu và thực nghiệm quá trình ứng dụng Differential privacy vào trong mô hình học sâu.
---------------------------------------------------------------------
Mô tả về các thư mục:
- data: tập dữ liệu sử dụng.
- html: lưu trữ các tệp html.
- model: chứa các tệp lưu mô hình.
- notebook: chứa các tệp jupyter notebook.
- report: chứa báo cáo của đồ án.
- static/image: chứa các hình ảnh sử dụng trong báo cáo. 
---------------------------------------------------------------------
Hướng dẫn sơ lược về cách sử dụng đồ án:
- Cài đặt jupyter notebook
- Cài đặt các thư viện cần thiết như torch, pandas, numpy, cv2,...
- !pip install opacus, !pip install adversarial-robustness-toolbox,...
- Có thể quan sát các bước xử lý dữ liệu, so sánh và chọn lọc mô hình đầy đủ nhất hãy mở thư mục notebook, lưu ý, độ chính xác giữa các lần chạy sẽ khác nhau. Có thể xem (không chạy) nhanh thông qua thư mục html, các tệp trong thư mục được mô tả như sau:
    - attack: quá trình tấn công mô hình học sâu.
    - attackdpmodel: quá trình tấn công mô hình học sâu có áp dụng DP.
    - build_model: quá trình xây dựng mô hình học sâu phân biệt ảnh thật và giả.
    - privacy: quá trình xây dựng mô hình học sâu có áp dụng Differential Privacy.
- Chạy ứng dụng streamlit thông qua câu lệnh streamlit run streamlit.py.
---------------------------------------------------------------------
Cảm ơn rất nhiều vì đã đọc file này. "# Differential-Privacy-In-Deep-Learning" 
