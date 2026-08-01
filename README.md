# Visual Product Similarity & Image-Based Recommendation System

## Project Overview

This project builds an image-based product recommendation system that retrieves visually similar products from a product image.

A user can either upload a new image or select a sample product from the dataset. The system converts the image into a learned visual embedding, searches the FAISS index using cosine similarity, and displays the Top-K most similar products.

The final application is presented through a Streamlit dashboard.

---

## Project Objective

The objective of this project is to:

- extract visual features from product images using a pretrained CNN;
- create dense image embeddings;
- index product embeddings using FAISS;
- retrieve visually similar products using cosine similarity;
- rank and display Top-K recommendations;
- evaluate retrieval quality using Precision@K and Recall@K;
- provide an interactive Streamlit interface for product search.

---

## Dataset

The project uses the **Stanford Online Products dataset**.

### Final selected dataset

- Total selected images: **3,095**
- Product classes: **600**
- Broad product categories: **12**
- Approximate images per product class: **5**

### Product categories

- Bicycle
- Cabinet
- Chair
- Coffee Maker
- Fan
- Kettle
- Lamp
- Mug
- Sofa
- Stapler
- Table
- Toaster

---

## Technology Stack

- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow / Keras
- **CNN Backbone:** EfficientNetB0
- **Image Processing:** Pillow and NumPy
- **Similarity Search:** FAISS
- **FAISS Index:** HNSW
- **Distance Metric:** Cosine Similarity
- **Data Handling:** Pandas
- **Frontend:** Streamlit

---

## Final System Workflow

```text
Input Product Image
        ↓
Convert to RGB
        ↓
Resize to 224 × 224
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
```

---

## Model Architecture

The final system uses EfficientNetB0 as the pretrained visual feature extractor.

### Training architecture

```text
Input Image
    ↓
EfficientNetB0 Backbone
    ↓
1280D Backbone Output
    ↓
512D Dense Projection Layer
    ↓
600-Class Product Classifier
```

### Retrieval architecture

After training, the final classifier layer is removed.

```text
Input Image
    ↓
EfficientNetB0 Backbone
    ↓
512D Retrieval Embedding
    ↓
L2 Normalization
    ↓
FAISS Search
```

The 512D projection does not replace EfficientNetB0. EfficientNetB0 remains the main CNN backbone, while the learned projection creates a smaller product-focused embedding for retrieval.

---

## Similarity Search

The final embeddings are normalized using L2 normalization.

FAISS uses inner-product search on normalized embeddings, which is equivalent to cosine similarity.

The final index uses:

- FAISS HNSW
- 512-dimensional embeddings
- cosine similarity
- Top-K retrieval

---

## Evaluation Metrics

### Final results

| Evaluation | Precision@3 | Description |
|---|---:|---|
| Baseline exact-product retrieval | 55.39% | Original 1280D EfficientNetB0 embeddings |
| Fine-tuned exact-product retrieval | 59.24% | Final 512D embeddings |
| Category-filtered exact-product retrieval | 63.21% | Exact-product search after category filtering |
| Category-level retrieval | 87.01% | Measures whether Top-3 results belong to the correct broad category |

### Search performance

- Average FAISS-only search time: **0.0588 ms**

### Metric definitions

- **Category Precision@3:** Measures how many of the Top-3 retrieved products belong to the same broad category as the query image.
- **Exact-Product Precision@3:** Measures how many Top-3 results match the query image's exact product class.
- **Filtered Exact-Product Precision@3:** Exact-product retrieval after applying a category filter.
- **Recall@3:** Measures how many relevant products were retrieved within the Top-3 results.
- **FAISS Search Time:** Measures only nearest-neighbor search time and does not include image loading or embedding extraction.

---

## Streamlit Dashboard

The Streamlit dashboard contains three main tabs.

### 1. Product Search

- Upload a new product image
- Select a sample image from the dataset
- Choose between 1 and 5 recommendations
- Generate a 512D query embedding
- Search the FAISS index
- Display Top-K visually similar products
- Show similarity score, category, class ID and image ID
- Display FAISS search time

### 2. Model Performance

- Category Precision@3
- Exact-Product Precision@3
- Filtered Exact-Product Precision@3
- Average FAISS search time
- Baseline versus fine-tuned comparison
- Category-wise performance chart
- Category-wise evaluation table

### 3. Project Details

- Project objective
- System workflow
- Model architecture
- Technology stack
- Dataset summary
- Evaluation metric explanations

---

## Project Structure

```text
Visual Product Similarity/
│
├── app/
│   └── visual_product_similarity_dashboard.py
│
├── data/
│   ├── Raw/
│   │   └── Stanford_Online_Products/
│   │       ├── bicycle_final/
│   │       ├── cabinet_final/
│   │       ├── chair_final/
│   │       ├── coffee_maker_final/
│   │       ├── fan_final/
│   │       ├── kettle_final/
│   │       ├── lamp_final/
│   │       ├── mug_final/
│   │       ├── sofa_final/
│   │       ├── stapler_final/
│   │       ├── table_final/
│   │       └── toaster_final/
│   │
│   └── Processed/
│       ├── selected_product_metadata.csv
│       ├── product_embeddings.npy
│       ├── product_faiss_index.index
│       ├── product_faiss_hnsw_index.index
│       ├── fine_tuned_product_classifier.keras
│       ├── fine_tuned_embedding_model.keras
│       ├── fine_tuned_product_embeddings.npy
│       ├── fine_tuned_product_faiss_hnsw_index.index
│       ├── category_evaluation_at_3.csv
│       └── category_wise_performance_at_3.csv
│
├── notebooks/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_faiss_indexing_and_retrieval.ipynb
│   ├── 04_baseline_evaluation.ipynb
│   └── 05_embedding_fine_tuning.ipynb
│
└── README.md
```

---

## Important Final Artifacts

The Streamlit dashboard uses these files:

```text
selected_product_metadata.csv
fine_tuned_embedding_model.keras
fine_tuned_product_embeddings.npy
fine_tuned_product_faiss_hnsw_index.index
category_wise_performance_at_3.csv
```

The final classifier file is not required for similarity retrieval because the dashboard uses the classifier-removed 512D embedding model.

---

## Installation

Install the required packages:

```bash
pip install streamlit tensorflow faiss-cpu pandas numpy pillow
```

On Windows, run this inside the same Python environment used for the notebooks.

---

## Run the Streamlit Dashboard

From the project root folder, run:

```bash
streamlit run app/visual_product_similarity_dashboard.py
```

The dashboard will open in the browser.

---

## How to Use

### Dataset sample mode

1. Open the Product Search tab.
2. Select **Select Dataset Sample**.
3. Choose a product category.
4. Choose a sample image.
5. Select the number of recommendations.
6. Click **Find Similar Products**.

The selected query image is excluded from its own recommendation results.

### Upload mode

1. Select **Upload New Image**.
2. Upload a JPG, JPEG or PNG image.
3. Select the number of recommendations.
4. Click **Find Similar Products**.

The model generates a new 512D embedding and searches all indexed products.

---

## Key Features

- EfficientNetB0-based visual feature extraction
- Fine-tuned 512D product embeddings
- Fast FAISS HNSW similarity retrieval
- Cosine-similarity ranking
- Top-K recommendation display
- Dataset sample and uploaded-image query modes
- Query image exclusion for dataset samples
- Category-level and exact-product evaluation
- Minimalist Streamlit dashboard

---

## Limitations

- Visual similarity does not always guarantee semantic identity.
- Products from different categories may look visually similar.
- Uploaded images with unusual backgrounds, angles or poor lighting may produce weaker recommendations.
- The system uses image information only and does not currently include price, availability, brand or textual metadata.

---

## Future Improvements

- Automatic category prediction for uploaded images
- Optional category filtering during search
- Brand, price and availability filters
- Larger product dataset
- More advanced metric-learning methods
- Cloud deployment
- User feedback and relevance-based reranking
- Product-detail pages and recommendation history

---

## Conclusion

This project demonstrates a complete visual product recommendation pipeline using deep learning and approximate nearest-neighbor search.

EfficientNetB0 extracts visual features, a learned 512D projection creates product-focused embeddings, and FAISS HNSW performs fast cosine-similarity retrieval. The final Streamlit dashboard makes the system interactive and allows users to retrieve visually similar products from either uploaded images or dataset samples.
