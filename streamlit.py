import os
import PIL.Image
import streamlit as st
import sys
# Import standard dependencies
import cv2
import os
import random
from matplotlib import pyplot as plt
sys.path.append("E:\\HKIIY4\\KLTN_20133100_20133019\\")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import torchvision 
from PIL import Image
import pandas as pd
from art.estimators.classification import  PyTorchClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased
import streamlit.components.v1 as components
import base64
from pathlib import Path
def intro():
    with open('./html/index.html', 'r', encoding='utf-8') as file:
        html_content = file.read()

# Mã hóa hình ảnh và thay thế placeholder trong HTML
    image_base64 = img_to_bytes('./static/image/logotruong.png')
    html_content = html_content.replace('{{IMAGE_PLACEHOLDER}}', image_base64)
    image_base64 = img_to_bytes('./static/image/logokhoa.png')
    html_content = html_content.replace('{{IMAGE_PLACEHOLDER1}}', image_base64)

    # Hiển thị nội dung HTML bằng Streamlit
    st.markdown(html_content, unsafe_allow_html=True)
    
    
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
# Lớp khoảng cách L1 của Siamese
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8192 , 512)  # Adjust input size to match flattened feature map size
        self.fc2 = nn.Linear(512, 2)  # Adjust output size to 2 for binary classification (fake vs real)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_history_data(path):
        with open(path, 'rb') as f:
            history = pickle.load(f)
        return history

def review():
    # Streamlit app
    st.title('Hàm mất mát và độ chính xác mô hình gốc')

    # Load history data
    history = load_history_data('./model/history.pkl')
    epochs = list(range(1, len(history['train_loss']) + 1))

    # Slider for epoch selection
    selected_epoch = st.slider('Select Epoch', min_value=1, max_value=len(epochs), value=len(epochs))

    # Plotting train loss and validation loss dynamically
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

    # Plot train and validation loss
    ax1.plot(epochs[:selected_epoch], history['train_loss'][:selected_epoch], label='Train Loss', marker='o')
    ax1.plot(epochs[:selected_epoch], history['val_loss'][:selected_epoch], label='Validation Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Train and Validation Losses (Epoch {selected_epoch})')
    ax1.legend()

    # Plot train and validation accuracy
    ax2.plot(epochs[:selected_epoch], history['train_acc'][:selected_epoch], label='Train Accuracy', marker='o')
    ax2.plot(epochs[:selected_epoch], history['val_acc'][:selected_epoch], label='Validation Accuracy', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Train and Validation Accuracy (Epoch {selected_epoch})')
    ax2.legend()

    # Adjust layout
    plt.tight_layout()

    # Display the plot using Streamlit
    st.pyplot(fig)
    
    st.title('Hàm mất mát và độ chính xác mô hình DP')
    history1 = load_history_data('./model/historyDP.pkl')
    epochs = list(range(1, len(history1['train_loss']) + 1))

    # Slider for epoch selection
    selected_epoch = st.slider('Select Epoch', min_value=1, max_value=len(epochs), value=len(epochs))

    # Plotting train loss and validation loss dynamically
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

    # Plot train and validation loss
    ax1.plot(epochs[:selected_epoch], history1['train_loss'][:selected_epoch], label='Train Loss', marker='o')
    ax1.plot(epochs[:selected_epoch], history1['val_loss'][:selected_epoch], label='Validation Loss', marker='o')
    ax1.plot(epochs[:selected_epoch], history1['epsilon'][:selected_epoch], label='epsilon', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Train and Validation Losses (Epoch {selected_epoch})')
    ax1.legend()

    # Plot train and validation accuracy
    ax2.plot(epochs[:selected_epoch], history1['train_acc'][:selected_epoch], label='Train Accuracy', marker='o')
    ax2.plot(epochs[:selected_epoch], history1['val_acc'][:selected_epoch], label='Validation Accuracy', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Train and Validation Accuracy (Epoch {selected_epoch})')
    ax2.legend()

    # Adjust layout
    plt.tight_layout()

    # Display the plot using Streamlit
    st.pyplot(fig)
    
    st.title('Ma trận nhầm lẫn mô hình gốc')
    #load and show image
    image = Image.open('./static/image/matrix.png')
    st.image(image, caption='Confusion Matrix', use_column_width=True)
    st.title('(ROC) Curve without DP')
    image = Image.open('./static/image/Curve.png')
    st.image(image, caption='Đường cong ROC', use_column_width=True)
    
    st.title('Ma trận nhầm lẫn mô hình DP')
    #load and show image
    image = Image.open('./static/image/matrixDP.png')
    st.image(image, caption='Confusion Matrix', use_column_width=True)
    st.title('(ROC) Curve with DP')
    image = Image.open('./static/image/CurveDP.png')
    st.image(image, caption='Đường cong ROC mô hình DP', use_column_width=True)
    

    # Hiển thị bảng dữ liệu trong Streamlit
    st.title('Bảng số liệu kiểm tra trên tập kiểm thử')
    image = Image.open('./static/image/table.png')
    st.image(image, use_column_width=True)
    
    # Hiển thị bảng dữ liệu trong Streamlit
    st.title('Bảng độ chính xác tấn công suy luận')
    image = Image.open('./static/image/table2.png')
    st.image(image, use_column_width=True)
def load_model(path):
    model = CNNModel()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model
def preprocess_image(image):
    image = np.array(image) 
    image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)  # Change from HWC to CHW
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return image

def predict(image, model):
    image = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        print(outputs)
    return 'This is a fake image' if predicted.item() == 1 else 'This is a real image', predicted
def predictDP(image,model):
    image = preprocess_image(image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    model.eval()
    with torch.no_grad():
        images= image.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu()
    return 'This is a fake image' if predicted.item() == 1 else 'This is a real image', predicted


def demo():
    st.markdown('# Phân biệt ảnh')  
    model = load_model('./model/FakeAndRealState.pth')
    uploaded_file = st.file_uploader("Upload a image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)
        st.image(image, caption='Uploaded image', use_column_width=True)
        st.write("")
        image = image.convert("RGB")
        prediction,_ = predict(image, model)
        st.write(f'Result: {prediction}')
        st.write('Dự đoán theo mô hình DP')
        model2 = torch.load('./model/ModelDP.pth')
        prediction,_ = predictDP(image, model)
        st.write(f'Result: {prediction}')
        
       

def demoAttack():
    from art.utils import to_categorical
    st.markdown('# Tấn công suy luận thành viên')
    model = load_model('./model/FakeAndRealState.pth')
    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(model, criterion, input_shape=(3, 256, 256), nb_classes=2)
    attack = MembershipInferenceBlackBoxRuleBased(classifier)
    #AttackDP
    model2 = torch.load('./model/ModelDP.pth')
    model2.eval()
    Classifier2= PyTorchClassifier(model2,criterion,[3,256,256],2)
    attack2 = MembershipInferenceBlackBoxRuleBased(Classifier2)
    #
    uploaded_file = st.file_uploader("Upload a image", type=["jpg", "jpeg", "png"])
    label_option = st.radio("Choose a fact label for image:", ('Real', 'Fake'))

    if label_option == 'Real':
        label = 0
    else:
        label = 1 

    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)
        st.image(image, caption='Uploaded image', use_column_width=True)
        st.write("")
        image = image.convert("RGB")
        image = preprocess_image(image)
        label = np.array([label])  
        label = torch.tensor(label) 
        label = to_categorical(label, nb_classes=2) 
        inferred_t = attack.infer(image, label,property=False)
        inferred_t2 = attack2.infer(image, label,property=False)
        st.write('The image not use for training processing' if 1 - inferred_t <0.4 else 'The image use for training processing')
        st.write('DP infer')
        st.write('The image not use for training processing' if 1- inferred_t2<0.4  else 'The image use for training processing')
def demoRealTime():
    cap = cv2.VideoCapture(0)
    model_path = './model/FakeAndRealState.pth'  # Replace with your model path
    model = load_model(model_path)
    
    # Placeholder to display the frame
    frame_placeholder = st.empty()
    
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            st.error('Failed to capture frame.')
            break
        
        # Predict using the model
        _,prediction = predict(frame, model)
        
        # Draw label on frame
        label = "Real" if prediction == 0 else "Fake"
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Display the resulting frame in Streamlit
        frame_placeholder.image(frame, channels='BGR')
        
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
# Main execution
if __name__ == '__main__':        
    st.set_page_config(page_title='Phân biệt hình ảnh thật/giả', page_icon=':camera:', layout='centered', initial_sidebar_state='auto')

    st.sidebar.markdown('Chọn một trong các tùy chọn bên dưới để bắt đầu.')
    
    page_names_to_functions = {
        'Giới thiệu': intro,
        'Xem xét số liệu':review,
        'Nhận diện ảnh thật':demo,
        'Nhận diện ảnh thời gian thực':demoRealTime,
        'Tấn công suy luận thành viên': demoAttack,
    }
    page_name = st.sidebar.radio('Điều hướng', list(page_names_to_functions.keys()))
    page_names_to_functions[page_name]()