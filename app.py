# ============================================================
# AI CANCER ANALYSIS SYSTEM
# COMPLETE STREAMLIT APPLICATION
# ============================================================

# ------------------------------------------------------------
# IMPORT LIBRARIES
# ------------------------------------------------------------

import json
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Cancer Analysis System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD CUSTOM CSS
# ============================================================

with open(
    "static/style.css",
    "r",
    encoding="utf-8"
) as file:

    css = file.read()


st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True
)


# ============================================================
# HTML HELPER
# ============================================================

def html(content):
    """
    Render HTML directly in Streamlit.
    """

    st.html(content)


# ============================================================
# NAVIGATION FUNCTION
# ============================================================

def navigate_to(page_name):
    """
    Change the current application page.
    """

    st.session_state["page"] = page_name


# ============================================================
# INITIALIZE NAVIGATION
# ============================================================

if "page" not in st.session_state:

    st.session_state["page"] = "🏠 Home"


# ============================================================
# PROJECT PATHS
# ============================================================

# Brain tumor model
BRAIN_MODEL_PATH = (
    "brain_tumor/model/brain_model_v3.keras"
)

# Breast cancer model
BREAST_MODEL_PATH = (
    "breast_cancer/model/"
    "breast_cancer_model_resnet50.keras"
)

# Brain medical information
BRAIN_INFO_PATH = (
    "medical_information/brain_information.json"
)

# Breast medical information
BREAST_INFO_PATH = (
    "medical_information/breast_information.json"
)


# ============================================================
# CLASS NAMES
# ============================================================

# Classes used by the 4-class brain model
BRAIN_CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# Classes used by the breast model
BREAST_CLASSES = [
    "Benign",
    "Malignant"
]


# ============================================================
# LOAD BRAIN MODEL
# ============================================================

@st.cache_resource
def load_brain_model():

    model = tf.keras.models.load_model(
        BRAIN_MODEL_PATH
    )

    return model


# ============================================================
# LOAD BREAST MODEL
# ============================================================

@st.cache_resource
def load_breast_model():

    model = tf.keras.models.load_model(
        BREAST_MODEL_PATH
    )

    return model


# ============================================================
# LOAD BRAIN MEDICAL INFORMATION
# ============================================================

@st.cache_data
def load_brain_information():

    with open(
        BRAIN_INFO_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        information = json.load(file)

    return information


# ============================================================
# LOAD BREAST MEDICAL INFORMATION
# ============================================================

@st.cache_data
def load_breast_information():

    with open(
        BREAST_INFO_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        information = json.load(file)

    return information


# ============================================================
# PREPROCESS BRAIN IMAGE
# ============================================================

def preprocess_brain_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize(
        (300, 300)
    )

    # Convert image to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREPROCESS BREAST IMAGE
# ============================================================

def preprocess_breast_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to ResNet50 input size
    image = image.resize(
        (224, 224)
    )

    # Convert image to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# BRAIN TUMOR PREDICTION
# ============================================================

def predict_brain(image):

    # Load trained model
    model = load_brain_model()

    # Preprocess image
    processed_image = preprocess_brain_image(
        image
    )

    # Generate prediction
    predictions = model.predict(
        processed_image,
        verbose=0
    )

    # Convert prediction to NumPy array
    predictions = np.asarray(
        predictions
    )

    # Find highest probability class
    predicted_index = int(
        np.argmax(predictions[0])
    )

    # Get class name
    predicted_class = BRAIN_CLASSES[
        predicted_index
    ]

    # Get confidence
    confidence = float(
        predictions[0][predicted_index]
    )

    # Generate probability dictionary
    probabilities = {}

    for i in range(
        len(BRAIN_CLASSES)
    ):

        probabilities[
            BRAIN_CLASSES[i]
        ] = float(
            predictions[0][i]
        ) * 100

    return (
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# BREAST CANCER PREDICTION
# ============================================================

def predict_breast(image):

    # Load trained model
    model = load_breast_model()

    # Preprocess image
    processed_image = preprocess_breast_image(
        image
    )

    # Generate prediction
    prediction = model.predict(
        processed_image,
        verbose=0
    )

    # Convert prediction to NumPy array
    prediction = np.asarray(
        prediction
    )

    # Get malignant probability
    malignant_probability = float(
        prediction[0][0]
    )

    # Calculate benign probability
    benign_probability = (
        1.0 -
        malignant_probability
    )

    # Determine prediction
    if malignant_probability >= 0.5:

        predicted_class = "Malignant"

        confidence = malignant_probability

    else:

        predicted_class = "Benign"

        confidence = benign_probability

    # Store probabilities
    probabilities = {

        "Benign":
            benign_probability * 100,

        "Malignant":
            malignant_probability * 100
    }

    return (
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# DISPLAY CLASS PROBABILITIES
# ============================================================

def display_probabilities(
    probabilities
):

    html("""
    <div class="section-title">
        Class Probabilities
    </div>
    """)

    # Display each class
    for class_name, probability in probabilities.items():

        # Convert class name for display
        display_name = (
            class_name
            .replace(
                "notumor",
                "No Tumor"
            )
            .title()
        )

        # Display class name
        st.write(
            f"**{display_name}**"
        )

        # Display progress bar
        st.progress(
            min(
                int(round(probability)),
                100
            )
        )

        # Display exact percentage
        st.caption(
            f"{probability:.2f}%"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # Logo and title
    html("""
    <div style="
        text-align:center;
        padding:20px 5px 25px 5px;
    ">

        <div style="
            font-size:48px;
            margin-bottom:10px;
        ">
            🧬
        </div>

        <h2 style="
            color:white;
            margin:0;
        ">
            AI Cancer Analysis
        </h2>

        <p style="
            color:#94a3b8;
            font-size:13px;
        ">
            AI-assisted medical image analysis
        </p>

    </div>
    """)


    st.divider()


    # Navigation menu
    st.radio(
        "Navigation",

        [
            "🏠 Home",
            "🧠 Brain Tumor Analysis",
            "🎀 Breast Cancer Analysis",
            "📚 Medical Information",
            "ℹ️ About the Project"
        ],

        key="page"
    )


    st.divider()


    # Sidebar information
    html("""
    <div style="
        color:#64748b;
        font-size:12px;
        line-height:1.6;
    ">

        <strong style="
            color:#94a3b8;
        ">
            Educational Project
        </strong>

        <br><br>

        This application demonstrates
        deep learning for medical image
        classification.

        <br><br>

        It is not a clinical diagnostic tool.

    </div>
    """)


# ============================================================
# GET CURRENT PAGE
# ============================================================

page = st.session_state["page"]


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    # --------------------------------------------------------
    # HERO SECTION
    # --------------------------------------------------------

    html("""
    <div class="hero">

        <div class="hero-badge">
            ARTIFICIAL INTELLIGENCE • MEDICAL IMAGE ANALYSIS
        </div>

        <h1 class="hero-title">
            AI <span>Cancer Analysis</span> System
        </h1>

        <p class="hero-description">
            An academic AI-powered platform designed to analyze
            medical images using deep learning models for brain
            tumor and breast cancer classification, together
            with explainable AI using Grad-CAM.
        </p>

    </div>
    """)


    # --------------------------------------------------------
    # MODULE TITLE
    # --------------------------------------------------------

    html("""
    <div class="section-title">
        Choose an Analysis Module
    </div>

    <div class="section-subtitle">
        Select the medical image analysis module.
    </div>
    """)


    # --------------------------------------------------------
    # MODULE CARDS
    # --------------------------------------------------------

    col1, col2 = st.columns(
        2,
        gap="large"
    )


    # ========================================================
    # BRAIN TUMOR CARD
    # ========================================================

    with col1:

        html("""
        <div class="module-card">

            <div class="module-icon">
                🧠
            </div>

            <div class="module-title">
                Brain Tumor Analysis
            </div>

            <div class="module-description">
                Analyze brain MRI images using a deep learning
                model trained for glioma, meningioma, pituitary
                tumor and no-tumor classification.
            </div>

        </div>
        """)


        # Brain navigation button
        st.button(
            "🧠 Select Brain Tumor Analysis",

            key="home_brain",

            use_container_width=True,

            on_click=navigate_to,

            args=(
                "🧠 Brain Tumor Analysis",
            )
        )


    # ========================================================
    # BREAST CANCER CARD
    # ========================================================

    with col2:

        html("""
        <div class="module-card">

            <div class="module-icon">
                🎀
            </div>

            <div class="module-title">
                Breast Cancer Analysis
            </div>

            <div class="module-description">
                Analyze breast histopathology images using a
                ResNet50 deep learning model trained for benign
                and malignant classification.
            </div>

        </div>
        """)


        # Breast navigation button
        st.button(
            "🎀 Select Breast Cancer Analysis",

            key="home_breast",

            use_container_width=True,

            on_click=navigate_to,

            args=(
                "🎀 Breast Cancer Analysis",
            )
        )


    # --------------------------------------------------------
    # SYSTEM FEATURES
    # --------------------------------------------------------

    html("""
    <div class="section-title">
        System Features
    </div>
    """)


    f1, f2, f3, f4 = st.columns(4)


    # Feature 1
    with f1:

        html("""
        <div class="feature-card">

            <div class="feature-icon">
                🤖
            </div>

            <div class="feature-title">
                Deep Learning
            </div>

            <div class="feature-text">
                CNN-based image classification using trained
                deep learning models.
            </div>

        </div>
        """)


    # Feature 2
    with f2:

        html("""
        <div class="feature-card">

            <div class="feature-icon">
                🔥
            </div>

            <div class="feature-title">
                Grad-CAM
            </div>

            <div class="feature-text">
                Visual explanations of regions influencing
                model predictions.
            </div>

        </div>
        """)


    # Feature 3
    with f3:

        html("""
        <div class="feature-card">

            <div class="feature-icon">
                📊
            </div>

            <div class="feature-title">
                Confidence
            </div>

            <div class="feature-text">
                Prediction confidence and class probabilities.
            </div>

        </div>
        """)


    # Feature 4
    with f4:

        html("""
        <div class="feature-card">

            <div class="feature-icon">
                📚
            </div>

            <div class="feature-title">
                Medical Information
            </div>

            <div class="feature-text">
                General educational information about
                classification categories.
            </div>

        </div>
        """)


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    html("""
    <div class="warning-box">

        ⚠️ <strong>Educational Use Only:</strong>

        This application is an academic demonstration of
        artificial intelligence for medical image analysis.
        It should not be used as a medical diagnosis.

    </div>
    """)


# ============================================================
# BRAIN TUMOR ANALYSIS PAGE
# ============================================================

elif page == "🧠 Brain Tumor Analysis":

    # Hero
    html("""
    <div class="hero">

        <div class="hero-badge">
            BRAIN MRI • EFFICIENTNETB3
        </div>

        <h1 class="hero-title">
            Brain Tumor <span>Analysis</span>
        </h1>

        <p class="hero-description">
            Upload a brain MRI image and let the trained
            EfficientNetB3 model classify the image.
        </p>

    </div>
    """)


    # Upload title
    html("""
    <div class="section-title">
        Upload Brain MRI
    </div>

    <div class="section-subtitle">
        Supported formats: JPG, JPEG and PNG
    </div>
    """)


    # File uploader
    uploaded_brain = st.file_uploader(
        "📤 Choose a brain MRI image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ],

        key="brain_upload"
    )


    # If image uploaded
    if uploaded_brain is not None:

        # Open uploaded image
        brain_image = Image.open(
            uploaded_brain
        )


        # Create columns
        image_col, result_col = st.columns(
            [1, 1],
            gap="large"
        )


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        with image_col:

            html("""
            <div class="section-title">
                Uploaded MRI
            </div>
            """)

            st.image(
                brain_image,
                use_container_width=True
            )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        with result_col:

            html("""
            <div class="section-title">
                AI Prediction
            </div>
            """)


            with st.spinner(
                "Analyzing brain MRI..."
            ):

                try:

                    (
                        predicted_class,
                        confidence,
                        probabilities
                    ) = predict_brain(
                        brain_image
                    )


                    # Format predicted class
                    display_class = (
                        predicted_class
                        .replace(
                            "notumor",
                            "No Tumor"
                        )
                        .title()
                    )


                    # Result card
                    html(f"""
                    <div class="result-card">

                        <div class="result-label">
                            PREDICTED CATEGORY
                        </div>

                        <div class="result-value">
                            {display_class}
                        </div>

                        <br>

                        <div class="result-label">
                            CONFIDENCE
                        </div>

                        <div class="result-value">
                            {confidence * 100:.2f}%
                        </div>

                    </div>
                    """)


                    # Status message
                    if predicted_class == "notumor":

                        st.success(
                            "The model classified this image "
                            "as No Tumor."
                        )

                    else:

                        st.warning(
                            f"The model classified this image "
                            f"as {display_class}."
                        )


                    # Display probabilities
                    display_probabilities(
                        probabilities
                    )


                    # ------------------------------------------------
                    # MEDICAL INFORMATION
                    # ------------------------------------------------

                    brain_info = (
                        load_brain_information()
                    )


                    information = (
                        brain_info.get(
                            predicted_class,
                            None
                        )
                    )


                    if information is not None:

                        html("""
                        <div class="section-title">
                            Educational Information
                        </div>
                        """)


                        # Description
                        if "description" in information:

                            html(f"""
                            <div class="info-box">

                                <h3>
                                    📖 About This Category
                                </h3>

                                <p>
                                    {information["description"]}
                                </p>

                            </div>
                            """)


                        # Risk factors
                        if "risk_factors" in information:

                            st.subheader(
                                "Possible Risk Factors"
                            )

                            for factor in information[
                                "risk_factors"
                            ]:

                                st.write(
                                    f"• {factor}"
                                )


                        # Treatment
                        if "treatment" in information:

                            st.subheader(
                                "General Treatment Information"
                            )

                            for treatment in information[
                                "treatment"
                            ]:

                                st.write(
                                    f"• {treatment}"
                                )


                except Exception as error:

                    st.error(
                        "An error occurred while running "
                        "the brain tumor model."
                    )

                    st.exception(error)


    # Disclaimer
    html("""
    <div class="warning-box">

        ⚠️ <strong>Important:</strong>

        This prediction is generated by an academic
        deep learning model and must not be interpreted
        as a medical diagnosis.

    </div>
    """)


# ============================================================
# BREAST CANCER ANALYSIS PAGE
# ============================================================

elif page == "🎀 Breast Cancer Analysis":

    # Hero
    html("""
    <div class="hero">

        <div class="hero-badge">
            HISTOPATHOLOGY • RESNET50
        </div>

        <h1 class="hero-title">
            Breast Cancer <span>Analysis</span>
        </h1>

        <p class="hero-description">
            Upload a breast histopathology image and let the
            trained ResNet50 model classify it as benign
            or malignant.
        </p>

    </div>
    """)


    # Upload title
    html("""
    <div class="section-title">
        Upload Histopathology Image
    </div>

    <div class="section-subtitle">
        Supported formats: JPG, JPEG and PNG
    </div>
    """)


    # File uploader
    uploaded_breast = st.file_uploader(
        "📤 Choose a breast histopathology image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ],

        key="breast_upload"
    )


    # If image uploaded
    if uploaded_breast is not None:

        # Open image
        breast_image = Image.open(
            uploaded_breast
        )


        # Create columns
        image_col, result_col = st.columns(
            [1, 1],
            gap="large"
        )


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        with image_col:

            html("""
            <div class="section-title">
                Uploaded Image
            </div>
            """)

            st.image(
                breast_image,
                use_container_width=True
            )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        with result_col:

            html("""
            <div class="section-title">
                AI Prediction
            </div>
            """)


            with st.spinner(
                "Analyzing histopathology image..."
            ):

                try:

                    (
                        predicted_class,
                        confidence,
                        probabilities
                    ) = predict_breast(
                        breast_image
                    )


                    # Result card
                    html(f"""
                    <div class="result-card">

                        <div class="result-label">
                            PREDICTED CATEGORY
                        </div>

                        <div class="result-value">
                            {predicted_class}
                        </div>

                        <br>

                        <div class="result-label">
                            CONFIDENCE
                        </div>

                        <div class="result-value">
                            {confidence * 100:.2f}%
                        </div>

                    </div>
                    """)


                    # Status
                    if predicted_class == "Benign":

                        st.success(
                            "The model classified this image "
                            "as Benign."
                        )

                    else:

                        st.warning(
                            "The model classified this image "
                            "as Malignant."
                        )


                    # Display probabilities
                    display_probabilities(
                        probabilities
                    )


                    # ------------------------------------------------
                    # MEDICAL INFORMATION
                    # ------------------------------------------------

                    breast_info = (
                        load_breast_information()
                    )


                    # Convert class to lowercase
                    info_key = (
                        predicted_class.lower()
                    )


                    information = (
                        breast_info.get(
                            info_key,
                            None
                        )
                    )


                    if information is not None:

                        html("""
                        <div class="section-title">
                            Educational Information
                        </div>
                        """)


                        # Description
                        if "description" in information:

                            html(f"""
                            <div class="info-box">

                                <h3>
                                    📖 About This Category
                                </h3>

                                <p>
                                    {information["description"]}
                                </p>

                            </div>
                            """)


                        # Risk factors
                        if "risk_factors" in information:

                            st.subheader(
                                "Possible Risk Factors"
                            )

                            for factor in information[
                                "risk_factors"
                            ]:

                                st.write(
                                    f"• {factor}"
                                )


                        # Treatment
                        if "treatment" in information:

                            st.subheader(
                                "General Treatment Information"
                            )

                            for treatment in information[
                                "treatment"
                            ]:

                                st.write(
                                    f"• {treatment}"
                                )


                except Exception as error:

                    st.error(
                        "An error occurred while running "
                        "the breast cancer model."
                    )

                    st.exception(error)


    # Disclaimer
    html("""
    <div class="warning-box">

        ⚠️ <strong>Important:</strong>

        This prediction is generated by an academic
        deep learning model and must not be interpreted
        as a medical diagnosis.

    </div>
    """)


# ============================================================
# MEDICAL INFORMATION PAGE
# ============================================================

elif page == "📚 Medical Information":

    # Hero
    html("""
    <div class="hero">

        <div class="hero-badge">
            EDUCATIONAL INFORMATION
        </div>

        <h1 class="hero-title">
            Medical <span>Information</span>
        </h1>

        <p class="hero-description">
            General educational information associated with
            the categories used by the AI models.
        </p>

    </div>
    """)


    # Load information
    brain_info = (
        load_brain_information()
    )

    breast_info = (
        load_breast_information()
    )


    # --------------------------------------------------------
    # BRAIN INFORMATION
    # --------------------------------------------------------

    html("""
    <div class="section-title">
        🧠 Brain Tumor Categories
    </div>
    """)


    for key, information in brain_info.items():

        # Make sure information is a dictionary
        if isinstance(
            information,
            dict
        ):

            title = information.get(
                "title",
                key.title()
            )

            description = information.get(
                "description",
                ""
            )

            html(f"""
            <div class="info-box">

                <h3>
                    {title}
                </h3>

                <p>
                    {description}
                </p>

            </div>
            """)


    # --------------------------------------------------------
    # BREAST INFORMATION
    # --------------------------------------------------------

    html("""
    <div class="section-title">
        🎀 Breast Cancer Categories
    </div>
    """)


    for key, information in breast_info.items():

        # Make sure information is a dictionary
        if isinstance(
            information,
            dict
        ):

            title = information.get(
                "title",
                key.title()
            )

            description = information.get(
                "description",
                ""
            )

            html(f"""
            <div class="info-box">

                <h3>
                    {title}
                </h3>

                <p>
                    {description}
                </p>

            </div>
            """)


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About the Project":

    # Hero
    html("""
    <div class="hero">

        <div class="hero-badge">
            FINAL YEAR B.TECH CSE PROJECT
        </div>

        <h1 class="hero-title">
            About the <span>Project</span>
        </h1>

        <p class="hero-description">
            AI Cancer Analysis System demonstrates the use
            of deep learning and explainable artificial
            intelligence for medical image classification.
        </p>

    </div>
    """)


    # Brain module
    html("""
    <div class="info-box">

        <h3>
            🧠 Brain Tumor Module
        </h3>

        <p>
            The brain module uses an EfficientNetB3-based
            deep learning model trained for classification
            of glioma, meningioma, pituitary tumor and
            no-tumor MRI images.
        </p>

    </div>
    """)


    # Breast module
    html("""
    <div class="info-box">

        <h3>
            🎀 Breast Cancer Module
        </h3>

        <p>
            The breast cancer module uses a ResNet50-based
            deep learning model for benign and malignant
            histopathology image classification.
        </p>

    </div>
    """)


    # Explainable AI
    html("""
    <div class="info-box">

        <h3>
            🔥 Explainable AI
        </h3>

        <p>
            Grad-CAM is incorporated into the project to
            provide visual explanations of image regions
            influencing model predictions.
        </p>

    </div>
    """)


    # Disclaimer
    html("""
    <div class="warning-box">

        ⚠️ <strong>Educational Disclaimer:</strong>

        This system is developed for academic and
        educational purposes. It is not a clinical
        diagnostic tool.

    </div>
    """)


# ============================================================
# FOOTER
# ============================================================

html("""
<div class="custom-footer">

    <div class="footer-title">
        AI Cancer Analysis System
    </div>

    Academic Deep Learning Project •
    Medical Image Analysis

</div>
""")