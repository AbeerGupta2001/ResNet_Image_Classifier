import streamlit as st
import onnxruntime as ort
from transformers import AutoImageProcessor
import numpy as np
from PIL import Image
import requests
from io import BytesIO

#Label for Binary classification
id2label = {
    0: "Cat",
    1: "Dog",
}

st.header("ResNet Binary Image Classifier")
st.write("Binary Image Classifier to classify cat and dog")

#Loading model and image processor
@st.cache_resource
def load_model_processor():
    image_processor = AutoImageProcessor.from_pretrained("image_onnx")
    session = ort.InferenceSession(
        "image_onnx/model.onnx",
        providers=["CPUExecutionProvider"]
    )
    return image_processor,session

image_processor,session = load_model_processor()

def fetch_url(url):
    try:
        res = requests.get(url)
        if res.status_code == 200:
            image = Image.open(BytesIO(res.content)).convert("RGB").resize((224,224))
            return image
    except:
        st.error("Invalid image url")



options = st.radio("# Select anyone option:",["**Upload image**","**Image url**"])

def predict_image(image):
    try:
        inputs = image_processor(image,return_tensors="np")
        data = {"pixel_values":inputs["pixel_values"]}
        logits = session.run(None,data)[0]
        pred = np.argmax(logits,axis=1)[0]
        st.success(f"Predicted Class: {id2label[pred]}")
    except:
        st.error("Error predicting the image")

if options == "**Upload image**":
    uploader_image = st.file_uploader("## Upload Image",type=["jpg","webp","png"],)
    if uploader_image:
        image = Image.open(uploader_image).convert("RGB").resize((224,224))
        st.image(image)
        if st.button("Classify"):
            predict_image(image)
else:
    image_url = st.text_input("Enter image link:")
    if st.button("Fetch image and classify"):
        url_image = fetch_url(image_url)
        if url_image:
            st.image(url_image)
            predict_image(url_image)



# if uploaded_image:
#     data_image = Image.open(uploaded_image).convert("RGB").resize((224,224))
#     st.image(data_image)
#     if st.button("Classify"):
