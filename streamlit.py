# import streamlit as st
# import torch
# from torchvision import transforms
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from MedViT import MedViT_small
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# # --- Setup ---
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = MedViT_small(num_classes=5).to(device)

# # Load weights
# model_path = "model_weights.pkl"
# state_dict = torch.load(model_path, map_location=device)
# model.load_state_dict(state_dict, strict=False)
# model.eval()

# # Image preprocessing
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
# ])

# # --- Streamlit UI ---
# st.set_page_config(page_title="Cancer Detection from CT Scan")
# st.title("🩺 CT Scan Cancer Detection")
# st.write("Upload a CT scan image to predict cancer severity and see a heatmap explanation.")

# uploaded_file = st.file_uploader("Upload a CT Scan Image", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess
#     input_tensor = transform(image).unsqueeze(0).to(device)

#     # Prediction
#     with torch.no_grad():
#         output = model(input_tensor)
#         predicted_class = torch.argmax(output, dim=1).item()
#         score = predicted_class + 1

#     # Interpretation
#     interpretation = {
#         1: """🟢 Class 1: No signs of cancer.
# The scan looks completely normal. There are no visible issues in your lungs.
# You are healthy based on the scan, and no further action is needed.
# Regular check-ups are still important to stay on top of your health.

# - ✅ Lungs appear healthy and clear.
# - ❌ No abnormal growths or spots found.
# - 🛑 No treatment or follow-up required.
# - 📅 Keep up with routine annual health check-ups.""",

#         2: """🟢 Class 2: Very low likelihood of cancer.
# The scan shows very minor changes that are almost certainly harmless.
# There's no reason to worry, but it's good to stay aware and monitor your health.
# No immediate action is needed, just routine checkups.

# - 🧘‍♂️ Minor changes are usually benign.
# - 📉 Very low chance of being cancerous.
# - 👨‍⚕️ No need for immediate tests or treatment.
# - 📋 Recommended to continue regular check-ups.""",

#         3: """🟡 Class 3: Unclear findings. Follow-up recommended.
# Some spots in the scan are not clearly normal or abnormal.
# It’s not something to panic about, but a follow-up scan is advised.
# Doctors may want to keep an eye on it over time to be sure.

# - ⚠️ Some findings are indeterminate (not clearly normal/abnormal).
# - 🕒 A follow-up scan is usually advised in a few months.
# - 🔍 Close monitoring is important to detect changes.
# - 🧑‍⚕️ Talk to your doctor for next steps and reassurance.""",

#         4: """🟠 Class 4: Suspicious. Detailed check advised.
# The scan shows patterns that may be linked to cancer.
# Further tests like a biopsy or specialist consultation are recommended.
# It’s important to act early to rule out or confirm anything serious.

# - ❗ Suspicious features need deeper investigation.
# - 🧪 May require biopsy, PET/CT scan, or specialist opinion.
# - ⏳ Early detection is critical for best outcomes.
# - 👨‍⚕️ Consult a specialist as soon as possible.""",

#         5: """🔴 Class 5: High risk. Immediate consultation needed.
# The scan has strong signs that may indicate cancer.
# You should consult a doctor or specialist as soon as possible.
# Early diagnosis and treatment can make a big difference.

# - 🚨 Highly suspicious findings detected.
# - 🔬 Strong indicators of possible malignancy.
# - 🏥 Requires urgent consultation and further diagnostic tests.
# - 🩺 Early medical action can be life-saving.""",
#     }[score]

#     st.subheader("Prediction:")
#     st.success(f"Predicted Class: {score}")
#     st.info(interpretation)

#     # Grad-CAM
#     input_tensor.requires_grad_()
#     target_layers = [model.norm]
#     cam = GradCAM(model=model, target_layers=target_layers)
#     grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(predicted_class)])[0]
#     rgb_img = np.array(image.resize((224, 224))) / 255.0
#     cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

#     # Show Grad-CAM
#     st.subheader("Grad-CAM Heatmap:")
#     st.image(cam_image, caption="Model Explanation", use_column_width=True)

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from MedViT import MedViT_small
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MedViT_small(num_classes=5).to(device)

# Load weights
model_path = "model_weights.pkl"
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# --- Streamlit UI ---
st.set_page_config(page_title="🩺 CT Scan Cancer Detection", layout="centered")
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .title {
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        color: #264653;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🩺 AI-Based CT Scan Cancer Detection</p>', unsafe_allow_html=True)
st.write("Upload a CT scan image to predict cancer severity and visualize risk regions.")

uploaded_file = st.file_uploader("📤 Upload CT Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        score = predicted_class + 1

    interpretation = {
        1: """🟢 **Class 1: No signs of cancer**

- ✅ Lungs appear healthy and clear.
- ❌ No abnormal growths or spots found.
- 🛑 No treatment or follow-up required.
- 📅 Keep up with routine annual check-ups.""",

        2: """🟢 **Class 2: Very low likelihood of cancer**

- 🧘‍♂️ Minor changes are usually benign.
- 📉 Very low chance of being cancerous.
- 👨‍⚕️ No need for immediate tests or treatment.
- 📋 Recommended to continue regular check-ups.""",

        3: """🟡 **Class 3: Unclear findings. Follow-up recommended**

- ⚠️ Some findings are indeterminate.
- 🕒 A follow-up scan is advised in a few months.
- 🔍 Close monitoring is important.
- 🧑‍⚕️ Talk to your doctor for reassurance.""",

        4: """🟠 **Class 4: Suspicious. Detailed check advised**

- ❗ Suspicious features need deeper investigation.
- 🧪 May require biopsy or PET/CT scan.
- ⏳ Early detection is critical.
- 👨‍⚕️ Consult a specialist soon.""",

        5: """🔴 **Class 5: High risk. Immediate consultation needed**

- 🚨 Highly suspicious findings detected.
- 🔬 Strong indicators of possible malignancy.
- 🏥 Requires urgent medical consultation.
- 🩺 Early action can be life-saving.""",
    }[score]

    # Grad-CAM Visualization
    input_tensor.requires_grad_()
    target_layers = [model.norm]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(predicted_class)])[0]
    rgb_img = np.array(image.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Layout: Input on left, Grad-CAM on right
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="📸 Uploaded CT Scan", use_column_width=True)
    with col2:
        st.image(cam_image, caption="🔥 Grad-CAM Heatmap", use_column_width=True)

    # Prediction & interpretation below both
    st.markdown("---")
    st.markdown(f"### 🧠 Predicted Class: {score}")
    st.info(interpretation)
    st.success("✅ Analysis Complete. Please consult a medical professional for further evaluation.")
