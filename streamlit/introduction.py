


import streamlit as st

st.title("Fine-Tuned ResNet Image Classification")

st.markdown(
        """
        This application showcases an **image classification system built using a fine-tuned ResNet architecture**,
        designed for accurate and efficient inference in real-world scenarios. The model leverages a pretrained
        ResNet backbone and is fine-tuned on a domain-specific dataset to adapt learned visual representations
        to the target classification task.

        By applying **transfer learning**, the model benefits from robust low-level and high-level features learned
        from large-scale image datasets, while additional task-specific training enables strong performance even
        with limited labeled data. The fine-tuning strategy focuses on optimizing the classification layers and,
        where applicable, selectively unfreezing deeper convolutional blocks to improve generalization.

        The Streamlit-based interface provides a **simple and interactive user experience**, allowing users to upload
        images and receive predictions in real time. The inference pipeline is designed with deployment efficiency
        in mind, making the system suitable for web applications, internal tools, and further optimization pipelines
        such as ONNX-based inference.

        Overall, this project demonstrates a **complete production-oriented deep learning workflow**, spanning
        model fine-tuning, optimization, and deployment using industry-standard tools.
        """
    )