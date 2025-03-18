import os
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from datetime import datetime
from database import DDRSQL
from navigation import make_sidebar
from timm import create_model
import torch.nn as nn
from captum.attr import Saliency
from captum.attr import visualization as viz
import seaborn as sns
import plotly.graph_objects as go
from captum.attr import IntegratedGradients



# Initialize Streamlit sidebar
make_sidebar()

# Database connection
db_uri = 'postgresql://postgres:102414@localhost:5432/project'
ddrsql = DDRSQL(db_uri)

# Define categories
categories = ["0:No_DR", "1:Mild", "2:Moderate", "3:Severe", "4:Proliferate_DR"]

# Directory to save images
save_directory = "G:/data_upload"
os.makedirs(save_directory, exist_ok=True)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = create_model(
    "swin_small_patch4_window7_224", 
    pretrained=False, 
    num_classes=5, 
    global_pool="avg"  # ✅ ใช้ global average pooling
)


    # ✅ ปรับ model head ให้เหมือนตอน training
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 5)  # Output 5 classes
    )

    # ✅ โหลด state_dict ที่ตรงกับ architecture
    state_dict = torch.load(r'G:\project\webapp_ddr\save_model\swin_dr_final.pth', map_location=device)
    model.load_state_dict(state_dict, strict=False)  # ใช้ strict=False ป้องกัน key ไม่ตรง
    model.to(device)
    model.eval()
    return model

# ✅ แปลงรูปภาพให้ได้ขนาด (3, 224, 224)
def preprocess_image(image):
    img = np.array(image)

    # แปลง grayscale → RGB
    if len(img.shape) == 2 or img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

    return img

def preprocess_func(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)  # แปลงจาก numpy → PIL Image

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = preprocess(img)

    # ✅ แก้บั๊ก: ถ้าเป็น grayscale ให้ duplicate เป็น 3 channels
    if img_tensor.shape[0] == 1:  
        img_tensor = img_tensor.repeat(3, 1, 1)

    return img_tensor

# ✅ ทำ Prediction ให้ได้ output ที่ถูกต้อง
def make_prediction(model, processed_img):
    with torch.no_grad():
        processed_img = processed_img.unsqueeze(0).to(device)  # เพิ่ม batch dimension
        outputs = model(processed_img)

        # ✅ แก้ไข: ถ้า output มี 4 มิติ (batch, H, W, C) → ลด H, W เหลือ 1
        if outputs.dim() == 4:
            outputs = outputs.mean(dim=[1, 2])  # ลด H และ W

        print("📌 Fixed Model Output Shape:", outputs.shape)  # ✅ เช็กว่ากลายเป็น (1, 5) หรือยัง

        probs = outputs.softmax(1).cpu().numpy().flatten()
        return probs


def interpret_prediction(model, processed_img, target):
    target_tensor = torch.tensor(target, dtype=torch.long, device=processed_img.device)  # Convert target to tensor

    processed_img = processed_img.unsqueeze(0)  # Add batch dimension
    processed_img.requires_grad = True  # Ensure gradients are tracked

    ig = IntegratedGradients(model)
    
    # Forward pass to get model outputs
    outputs = model(processed_img)  # Shape: [1, num_classes]
    
    # Ensure we take the gradient w.r.t. the specific target class
    attributions = ig.attribute(processed_img, target=target_tensor.item())  

    return attributions.squeeze().cpu().detach().numpy()



# ✅ Streamlit UI
st.markdown("<h1 style='text-align: center;'>🩺 Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("👤 Patient Information")
    first_name = st.text_input("First Name", max_chars=20)
    last_name = st.text_input("Last Name", max_chars=20)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    diabetes_type = st.selectbox("Diabetes Type", ["Type 1", "Type 2"])
    diabetes_duration = st.number_input("Diabetes Duration (years)", min_value=0.0, step=0.1)
    hba1c_level = st.number_input("HbA1c Level", min_value=0.0, step=0.1)
    blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0.0, step=0.1)
    weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1)

with col2:
    st.subheader("📤 Upload Image")
    upload = st.file_uploader("Upload an Image:", type=["png", "jpg", "jpeg"])
    if upload:
        img = Image.open(upload)
        st.image(img, caption="Uploaded Image", width=250)

st.markdown("<hr>", unsafe_allow_html=True)
if upload:
    img = Image.open(upload)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        model = load_model()
        preprocessed_img = preprocess_func(preprocess_image(img))  # ✅ Trim and preprocess before prediction
        probs = make_prediction(model, preprocessed_img)  # ✅ ทำนาย
        y_classes = int(np.argmax(probs))  # ✅ Convert to int

        if len(probs) == len(categories):
            predicted_class = categories[y_classes]
            confidence = probs[y_classes]
            st.success(f"🔍 Prediction: {predicted_class} (Confidence: {confidence:.2f})")

            # Visualization for top predictions
            idxs = np.argsort(probs)[-5:]
            top_probs = probs[idxs]
            top_labels = np.array(categories)[idxs]
            
            fig = go.Figure(go.Bar(
                x=top_probs[::-1],
                y=top_labels[::-1],
                orientation='h',
                marker=dict(color=['dodgerblue']*4 + ['tomato'])
            ))
            fig.update_layout(
                title="Top 5 Class Predictions",
                xaxis_title="Probability",
                yaxis_title="Class",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        
        else:
            st.error(f"❌ Prediction output shape mismatch! Expected {len(categories)}, but got {len(probs)}")




