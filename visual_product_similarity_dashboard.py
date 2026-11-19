from pathlib import Path
import time

import faiss
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Visual Product Similarity Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

processed_data_dir = (
    project_root
    / "data"
    / "processed"
)

raw_data_dir = (
    project_root
    / "data"
    / "Raw"
    / "Stanford_Online_Products"
)


metadata_path = (
    processed_data_dir
    / "selected_product_metadata.csv"
)

embedding_model_path = (
    processed_data_dir
    / "fine_tuned_embedding_model.keras"
)

embeddings_path = (
    processed_data_dir
    / "fine_tuned_product_embeddings.npy"
)

faiss_index_path = (
    processed_data_dir
    / "fine_tuned_product_faiss_hnsw_index.index"
)

category_performance_path = (
    processed_data_dir
    / "category_wise_performance_at_3.csv"
)


# --------------------------------------------------
# Data and Model Loading Functions
# --------------------------------------------------

@st.cache_data
def load_metadata(metadata_file_path):
    metadata = pd.read_csv(metadata_file_path)
    return metadata

@st.cache_data
def load_category_performance(category_performance_file_path):
    category_performance = pd.read_csv(
        category_performance_file_path
    )
    return category_performance


@st.cache_data
def load_embeddings(embeddings_file_path):
    embeddings = np.load(embeddings_file_path)
    return embeddings


@st.cache_resource
def load_embedding_model(model_file_path):
    model = tf.keras.models.load_model(
        model_file_path,
        compile=False
    )
    return model


@st.cache_resource
def load_faiss_index(index_file_path):
    faiss_index = faiss.read_index(
        str(index_file_path)
    )
    return faiss_index


def preprocess_query_image(query_image):
    resized_image = query_image.resize((224, 224))
    image_array = np.array(resized_image, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)

    return image_batch


def extract_query_embedding(query_image, model):
    image_batch = preprocess_query_image(query_image)

    query_embedding = model.predict(image_batch, verbose=0)
    query_embedding = query_embedding.astype(np.float32)

    faiss.normalize_L2(query_embedding)

    return query_embedding


def search_similar_products(
    query_embedding,
    faiss_search_index,
    number_of_results,
    excluded_index=None
):
    additional_result = 1 if excluded_index is not None else 0

    similarity_scores, result_indices = faiss_search_index.search(
        query_embedding,
        number_of_results + additional_result
    )

    similarity_scores = similarity_scores[0]
    result_indices = result_indices[0]

    search_results = []

    for similarity_score, result_index in zip(similarity_scores, result_indices):
        if result_index == excluded_index:
            continue

        search_results.append({
            "result_index": int(result_index),
            "similarity_score": float(similarity_score)
        })

        if len(search_results) == number_of_results:
            break

    return search_results

# --------------------------------------------------
# Load Processed Project Artifacts
# --------------------------------------------------

product_metadata = load_metadata(
    metadata_path
)

category_performance = load_category_performance(
    category_performance_path
)

fine_tuned_embeddings = load_embeddings(
    embeddings_path
)

embedding_model = load_embedding_model(
    embedding_model_path
)

faiss_index = load_faiss_index(
    faiss_index_path
)


# --------------------------------------------------
# Prepare Product Image Paths
# --------------------------------------------------

product_metadata["image_path"] = product_metadata[
    "path"
].apply(
    lambda relative_path: raw_data_dir / Path(relative_path)
)


category_options = sorted(
    product_metadata["category_name"]
    .dropna()
    .unique()
    .tolist()
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("🛍️ VisualMatch")
st.sidebar.caption("Visual Product Intelligence")

st.sidebar.divider()

st.sidebar.caption(
    "Fine-tuned EfficientNetB0 with "
    "FAISS-powered similarity search."
)

st.sidebar.divider()

st.sidebar.subheader("System Information")

st.sidebar.metric(
    label="Products Indexed",
    value=f"{len(product_metadata):,}"
)

st.sidebar.metric(
    label="Embedding Dimension",
    value=f"{fine_tuned_embeddings.shape[1]}D"
)

st.sidebar.caption("Backbone: EfficientNetB0")
st.sidebar.caption("Backbone Output: 1280D")
st.sidebar.caption("Final Retrieval Embedding: 512D")
st.sidebar.caption("Index: FAISS HNSW")
st.sidebar.caption("Similarity: Cosine Similarity")

st.sidebar.divider()

st.sidebar.caption("GUVI × HCL Capstone")
st.sidebar.caption(
    "Domain: Computer Vision & Recommendation"
)


# --------------------------------------------------
# Main Header
# --------------------------------------------------

st.title("Visual Product Similarity Search")

st.caption(
    "Upload or select a product image and retrieve visually "
    "similar products using deep-learning embeddings and FAISS."
)

st.divider()


# --------------------------------------------------
# Main Navigation Tabs
# --------------------------------------------------

search_tab, performance_tab, details_tab = st.tabs(
    [
        "🔍 Product Search",
        "📈 Model Performance",
        "ℹ️ Project Details"
    ]
)


# --------------------------------------------------
# Product Search Tab
# --------------------------------------------------

with search_tab:
    st.subheader("Find Similar Products")

    query_source = st.radio(
        "Choose Query Source",
        options=["Upload New Image", "Select Dataset Sample"],
        horizontal=True
    )

    st.divider()

    query_image = None
    query_dataset_index = None
    query_category = None

    if query_source == "Upload New Image":
        uploaded_image = st.file_uploader(
            "Upload a product image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            query_image = Image.open(uploaded_image).convert("RGB")

            st.image(
                query_image,
                caption="Uploaded Query Image",
                width=300
            )
        else:
            st.info(
                "Upload a JPG, JPEG or PNG product image "
                "to begin the similarity search."
            )

    else:
        selected_category = st.selectbox(
            "Select Product Category",
            options=category_options,
            format_func=lambda category: category.replace("_", " ").title()
        )

        category_products = product_metadata[
            product_metadata["category_name"] == selected_category
        ].copy()

        category_products = category_products.reset_index()

        selected_product_position = st.selectbox(
            "Select Sample Image",
            options=range(len(category_products)),
            format_func=lambda position: f"Image {position + 1}"
        )

        selected_product = category_products.iloc[selected_product_position]

        query_dataset_index = int(selected_product["index"])
        query_category = selected_product["category_name"]

        query_image = Image.open(
            selected_product["image_path"]
        ).convert("RGB")

        st.image(
            query_image,
            caption=query_category.replace("_", " ").title(),
            width=300
        )

    if query_image is not None:
        st.divider()

        top_k = st.slider(
            "Number of Similar Products",
            min_value=1,
            max_value=5,
            value=3,
            step=1
        )

        search_button = st.button("🔍 Find Similar Products")

        if search_button:
            with st.spinner("Searching for visually similar products..."):
                query_embedding = extract_query_embedding(
                    query_image=query_image,
                    model=embedding_model
                )

                search_start_time = time.perf_counter()

                search_results = search_similar_products(
                    query_embedding=query_embedding,
                    faiss_search_index=faiss_index,
                    number_of_results=top_k,
                    excluded_index=query_dataset_index
                )

                search_end_time = time.perf_counter()
                faiss_search_time_ms = (
                    search_end_time - search_start_time
                ) * 1000

            st.success("Similarity search completed successfully.")

            st.divider()

            st.subheader(
                f"Top-{len(search_results)} Similar Products"
            )

            result_columns = st.columns(len(search_results))

            for rank, search_result in enumerate(search_results, start=1):
                result_index = search_result["result_index"]
                similarity_score = search_result["similarity_score"]

                result_product = product_metadata.iloc[result_index]

                result_image = Image.open(
                    result_product["image_path"]
                ).convert("RGB")

                with result_columns[rank - 1]:
                    st.caption(f"Rank {rank}")

                    st.image(
                        result_image,
                        use_container_width=True
                    )

                    st.metric(
                        label="Similarity Score",
                        value=f"{similarity_score:.4f}"
                    )

                    st.write(
                        "**Category:**",
                        result_product["category_name"]
                        .replace("_", " ")
                        .title()
                    )

                    st.write(
                        "**Class ID:**",
                        int(result_product["class_id"])
                    )

                    st.write(
                        "**Image ID:**",
                        int(result_product["image_id"])
                    )

            st.divider()

            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

            summary_col1.metric(
                "Products Searched",
                f"{faiss_index.ntotal:,}"
            )

            summary_col2.metric(
                "Recommendations",
                len(search_results)
            )

            summary_col3.metric(
                "Embedding Dimension",
                f"{query_embedding.shape[1]}D"
            )

            summary_col4.metric(
                "FAISS Search Time",
                f"{faiss_search_time_ms:.4f} ms"
            )


# --------------------------------------------------
# Model Performance Tab
# --------------------------------------------------

with performance_tab:
    st.subheader("Model Performance Overview")

    st.info(
        "This section summarizes the retrieval quality and "
        "search efficiency of the final visual similarity system."
    )

    st.divider()
    st.subheader("Key Performance Indicators")

    performance_col1, performance_col2, performance_col3, performance_col4 = st.columns(4)

    performance_col1.metric(
        label="Category Precision@3",
        value="86.67%"
    )

    performance_col2.metric(
        label="Exact-Product Precision@3",
        value="59.46%"
    )

    performance_col3.metric(
        label="Filtered Exact-Product Precision@3",
        value="63.28%"
    )

    performance_col4.metric(
        label="Average FAISS Search Time",
        value="≈0.0549 ms"
    )

    st.divider()
    st.subheader("Baseline vs Fine-Tuned Performance")

    performance_summary = pd.DataFrame({
        "Model Stage": [
            "Baseline EfficientNetB0",
            "Fine-Tuned Embedding",
            "Category-Filtered Retrieval",
            "Category-Level Retrieval"
        ],
        "Embedding": [
            "1280D",
            "512D",
            "512D",
            "512D"
        ],
        "Evaluation Type": [
            "Exact Product",
            "Exact Product",
            "Exact Product",
            "Product Category"
        ],
        "Precision@3": [
            0.5539,
            0.5946,
            0.6328,
            0.8667
        ]
    })

    performance_summary["Precision@3"] = (
        performance_summary["Precision@3"] * 100
    ).round(2).astype(str) + "%"

    st.dataframe(
        performance_summary,
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader("Category-Wise Precision@3")

    category_precision_chart = category_performance[
        ["category_name", "average_category_precision_at_3"]
    ].copy()

    category_precision_chart["average_category_precision_at_3"] *= 100

    category_precision_chart["category_name"] = (
        category_precision_chart["category_name"]
        .str.replace("_", " ")
        .str.title()
    )

    category_precision_chart = (
        category_precision_chart
        .sort_values(
            by="average_category_precision_at_3",
            ascending=False
        )
        .set_index("category_name")
    )

    st.bar_chart(
        category_precision_chart,
        horizontal=True,
        use_container_width=True
    )

    st.divider()
    st.subheader("Category Performance Details")

    category_performance_table = category_performance.copy()

    category_performance_table["category_name"] = (
        category_performance_table["category_name"]
        .str.replace("_", " ")
        .str.title()
    )

    category_performance_table["average_category_precision_at_3"] = (
        category_performance_table["average_category_precision_at_3"] * 100
    ).round(2).astype(str) + "%"

    category_performance_table["average_category_recall_at_3"] = (
        category_performance_table["average_category_recall_at_3"] * 100
    ).round(2).astype(str) + "%"

    category_performance_table = category_performance_table.rename(
        columns={
            "category_name": "Category",
            "number_of_queries": "Queries",
            "average_category_precision_at_3": "Category Precision@3",
            "average_category_recall_at_3": "Category Recall@3"
        }
    )

    st.dataframe(
        category_performance_table,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Category Precision@3 measures how many of the Top-3 "
        "retrieved products belong to the same broad category "
        "as the query image."
    )

    st.caption(
        "Exact-product evaluation is stricter because a result "
        "must match the query's exact product class_id."
    )


# --------------------------------------------------
# Project Details Tab
# --------------------------------------------------

with details_tab:
    st.subheader("Project Details")

    st.info(
        "This project retrieves visually similar products using "
        "deep-learning image embeddings and FAISS similarity search."
    )

    st.divider()
    st.subheader("Project Objective")

    st.write(
        "The objective is to build an image-based recommendation "
        "system that accepts a product image and returns the most "
        "visually similar products from the dataset."
    )

    st.divider()
    st.subheader("System Workflow")

    workflow_col1, workflow_col2, workflow_col3 = st.columns(3)

    with workflow_col1:
        st.caption("Input and Feature Extraction")
        st.write(
            """
            1. Product image is selected or uploaded  
            2. Image is converted to RGB  
            3. Image is resized to 224 × 224  
            4. EfficientNetB0 extracts visual features
            """
        )

    with workflow_col2:
        st.caption("Embedding Generation")
        st.write(
            """
            1. EfficientNetB0 produces 1280D features  
            2. Learned projection converts them to 512D  
            3. The final classifier is removed  
            4. The 512D vector becomes the retrieval embedding
            """
        )

    with workflow_col3:
        st.caption("Similarity Retrieval")
        st.write(
            """
            1. Query embedding is L2-normalized  
            2. FAISS HNSW searches the indexed embeddings  
            3. Cosine similarity scores are calculated  
            4. Top-K similar products are displayed
            """
        )

    st.divider()
    st.subheader("Model Architecture")

    architecture_table = pd.DataFrame({
        "Component": [
            "Input Image",
            "Pretrained Backbone",
            "Backbone Output",
            "Projection Layer",
            "Final Retrieval Embedding",
            "Similarity Index"
        ],
        "Configuration": [
            "224 × 224 × 3",
            "EfficientNetB0",
            "1280D",
            "Dense 512D",
            "512D normalized vector",
            "FAISS HNSW"
        ],
        "Purpose": [
            "Standardized model input",
            "Extract visual product features",
            "General visual representation",
            "Learn product-focused features",
            "Used for similarity search",
            "Fast approximate nearest-neighbor retrieval"
        ]
    })

    st.dataframe(
        architecture_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader("Technology Stack")

    technology_col1, technology_col2 = st.columns(2)

    with technology_col1:
        st.write(
            """
            **Programming Language:** Python  
            **Deep Learning:** TensorFlow / Keras  
            **CNN Model:** EfficientNetB0  
            **Image Processing:** Pillow and NumPy
            """
        )

    with technology_col2:
        st.write(
            """
            **Similarity Search:** FAISS  
            **Index Type:** HNSW  
            **Distance Metric:** Cosine Similarity  
            **Frontend:** Streamlit
            """
        )

    st.divider()
    st.subheader("Dataset Summary")

    dataset_col1, dataset_col2, dataset_col3, dataset_col4 = st.columns(4)

    dataset_col1.metric(
        label="Selected Images",
        value=f"{len(product_metadata):,}"
    )

    dataset_col2.metric(
        label="Product Classes",
        value=f"{product_metadata['class_id'].nunique():,}"
    )

    dataset_col3.metric(
        label="Categories",
        value=f"{product_metadata['category_name'].nunique():,}"
    )

    dataset_col4.metric(
        label="Images per Class",
        value="Approximately 5"
    )

    st.divider()
    st.subheader("Product Categories")

    category_names = [
        category.replace("_", " ").title()
        for category in category_options
    ]

    st.write(", ".join(category_names))

    st.divider()
    st.subheader("Evaluation Metrics")

    metric_table = pd.DataFrame({
        "Metric": [
            "Category Precision@3",
            "Exact-Product Precision@3",
            "Filtered Exact-Product Precision@3",
            "Recall@3",
            "FAISS Search Time"
        ],
        "Meaning": [
            "Measures how many Top-3 results belong to the correct category",
            "Measures how many Top-3 results match the exact product class",
            "Exact-product retrieval after category filtering",
            "Measures how many relevant products were retrieved",
            "Measures only the nearest-neighbor search duration"
        ]
    })

    st.dataframe(
        metric_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader("Final Retrieval Pipeline")

    st.code(
        """
Input Product Image
        ↓
RGB Conversion and Resize to 224 × 224
        ↓
Fine-Tuned EfficientNetB0
        ↓
1280D Backbone Features
        ↓
512D Learned Projection
        ↓
L2 Normalization
        ↓
FAISS HNSW Cosine Search
        ↓
Top-K Visually Similar Products
        """,
        language="text"
    )