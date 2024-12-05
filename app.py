import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

# Set Omori-inspired styles
st.markdown(
    """
    <style>
        body {
            background-color: #1a1a1a;  /* Black background */
            color: #ffffff;  /* White text */
        }
        .title {
            color: #ff007f;  /* Vibrant pink for titles */
            font-size: 36px;
            font-weight: bold;
        }
        .subheader {
            color: #7f00ff;  /* Purple for subheaders */
            font-size: 24px;
        }
        .file-upload {
            color: #ffffff;
            font-size: 18px;
        }
        .prediction {
            color: #ff4d4d;  /* Red for prediction results */
            font-size: 20px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Function for MobileNetV2 ImageNet model
def mobilenetv2_imagenet():
    st.markdown('<p class="title">Image Classification with MobileNetV2</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Upload an image to classify it using MobileNetV2 trained on ImageNet</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying...")
        
        # Load MobileNetV2 model
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        
        # Preprocess the image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Make predictions
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
        
        st.markdown('<p class="subheader">Predictions:</p>', unsafe_allow_html=True)
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.markdown(f'<p class="prediction">{label}: {score * 100:.2f}%</p>', unsafe_allow_html=True)

# Function for CIFAR-10 model
def cifar10_classification():
    st.markdown('<p class="title">CIFAR-10 Image Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Upload an image to classify it into one of the 10 CIFAR-10 categories</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying...")
        
        # Load CIFAR-10 model
        model = tf.keras.models.load_model('cifar10_model.h5')
        
        # CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Preprocess the image
        img = image.resize((32, 32))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        st.markdown('<p class="subheader">Predicted Class:</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="prediction">{class_names[predicted_class]} ({confidence * 100:.2f}% confidence)</p>', unsafe_allow_html=True)

def flowers_classification():
    dataset_name = "tf_flowers"
    _, ds_info = tfds.load(dataset_name, split=["train"], as_supervised=True, with_info=True)
    class_names = ds_info.features["label"].names

    st.markdown('<p class="title">Flowers Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Upload an image to classify it into one of the flower categories</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying...")
        
        # Load the Flowers classification model
        model = tf.keras.models.load_model('mobilenetv2_tf_flowers.h5')
        

        # TF Flowers class names
        # class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']  # Replace with your actual class names if different
        
        # Preprocess the image
        img = image.resize((224, 224))  # Ensure size matches the model input
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        st.markdown('<p class="subheader">Predicted Class:</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="prediction">{class_names[predicted_class]} ({confidence * 100:.2f}% confidence)</p>', unsafe_allow_html=True)


# Main function to control the navigation
def main():
    st.sidebar.markdown(
        """
        <style>
            .sidebar-text {
                color: #ffffff;
                font-size: 20px;
                font-weight: bold;
            }
            .sidebar-item {
                color: #ff007f;
                font-size: 18px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown('<p class="sidebar-text">Navigation</p>', unsafe_allow_html=True)
    choice = st.sidebar.selectbox('<p class="sidebar-item">Choose Model</p>', ("CIFAR-10", "MobileNetV2 (ImageNet)","FlowerClassifier"), label_visibility="collapsed")
    
    if choice == "MobileNetV2 (ImageNet)":
        mobilenetv2_imagenet()
    elif choice == "CIFAR-10":
        cifar10_classification()
    elif choice == "FlowerClassifier":
        flowers_classification()

if __name__ == "__main__":
    main()
