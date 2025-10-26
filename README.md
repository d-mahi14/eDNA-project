# 🧬 eDNA Biodiversity Analysis Pipeline  
> **Version 4.7 — Deep Learning–Driven Environmental DNA (eDNA) Taxonomic Clustering and Diversity Analysis**

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20API-green?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-VAE-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Build-Stable-success)

---

## 🌿 Overview

The **eDNA Biodiversity Analysis Pipeline** provides an end-to-end workflow for **DNA-based biodiversity discovery**.  
It simulates, embeds, clusters, and analyzes **environmental DNA sequences (eDNA)** using a **Variational Autoencoder (VAE)** and **HDBSCAN** clustering — wrapped inside a user-friendly **Flask web API**.

---

## ✨ Key Features

✅ **Mock Sequence Generator** — Create realistic eDNA datasets with mutations and random variation  
✅ **K-mer Encoding** — Convert raw sequences into normalized frequency vectors  
✅ **Deep Latent Embedding (VAE)** — Learn compressed, noise-tolerant DNA representations  
✅ **HDBSCAN Clustering** — Detect taxonomic groups without fixed cluster numbers  
✅ **Biodiversity Metrics** — Compute Shannon, Simpson, Pielou, and species richness indices  
✅ **Partial Taxonomy for Noise** — Infer likely taxa for unclustered sequences  
✅ **UMAP Visualization** — Interactive 2D plots of latent embeddings  
✅ **Flask Web API** — Easily integrate into web or bioinformatics workflows

---

## 🧩 Repository Structure

```
📦 eDNA-Biodiversity
├── analysis.py          # Core ML and analysis pipeline
├── main.py              # Flask web service exposing endpoints
├── templates/
│   └── index.html       # Optional front-end page for testing
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/eDNA-Biodiversity.git
cd eDNA-Biodiversity
```

### 2️⃣ Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
python3 -m pip install -r requirements.txt
```
If you don’t have a `requirements.txt`, install manually:
```bash
python3 -m pip install flask tensorflow scikit-learn numpy matplotlib umap-learn hdbscan
```

---

## 🧠 Running the Application

### ▶️ Start the Server
```bash
python3 main.py
```
Access the web app at:  
🌐 **http://localhost:5000**

---

## 🔌 API Endpoints

### **`GET /generate_mock_data`**
Generates 1000 mock DNA sequences for testing.

**Response Example**
```json
{
  "sequences": ">sample_sequence_1\nATGCTAG...\n>sample_sequence_2\nTGCATGA...\n..."
}
```

---

### **`POST /analyze`**
Runs the complete biodiversity pipeline.

**Request JSON**
```json
{
  "sequence": ">seq1\nATGCTAGCTAG...\n>seq2\nCGTATCGT...\n..."
}
```

**Response JSON**
```json
{
  "summary": {
    "biodiversity_metrics": {
      "species_richness": 42,
      "shannon_diversity_index": 2.8765,
      "simpson_diversity_index": 0.9132,
      "pielou_evenness_index": 0.8231
    },
    "taxonomic_summary": {
      "identified_taxa": 36,
      "novel_taxa": 6,
      "noise_points": 12
    },
    "taxa_details": [...],
    "noise_point_details": [...]
  },
  "plots": {
    "cluster_plot": "<base64_image>"
  }
}
```

---

## 🧬 Pipeline Flow

| Step | Description | Function |
|------|--------------|-----------|
| 1️⃣ | Generate mock or input eDNA sequences | `generate_mock_edna_sequences()` |
| 2️⃣ | Convert sequences into 4-mer vectors | `sequences_to_kmers()` |
| 3️⃣ | Train VAE for latent embeddings | `train_vae_and_get_embeddings()` |
| 4️⃣ | Cluster embeddings using HDBSCAN | `run_hdbscan()` |
| 5️⃣ | Compute biodiversity metrics | `analyze_biodiversity()` |
| 6️⃣ | Save 2D UMAP visualization | `save_cluster_scatter()` |

---

## 📊 Example Usage (Direct Script)

```python
import analysis

# Step 1: Generate mock eDNA data
sequences = analysis.generate_mock_edna_sequences(500)

# Step 2: Convert to k-mer vectors
kmer_vectors = analysis.sequences_to_kmers(sequences)

# Step 3: Train VAE and extract embeddings
embeddings = analysis.train_vae_and_get_embeddings(kmer_vectors)

# Step 4: Cluster with HDBSCAN
labels = analysis.run_hdbscan(embeddings)

# Step 5: Analyze biodiversity
summary = analysis.analyze_biodiversity(labels, embeddings, sequences)

print(summary['biodiversity_metrics'])
```

---

## 📈 Example Output

```
--- Starting VAE Training ---
--- VAE Training and Embedding Extraction Complete ---
--- Running HDBSCAN (min_cluster_size=10) ---
HDBSCAN found 8 clusters and 12 noise points.
--- Analyzing Biodiversity ---
Biodiversity analysis complete.

Biodiversity Metrics:
 - Species Richness: 42
 - Shannon Index: 2.87
 - Simpson Index: 0.91
 - Pielou Evenness: 0.82
```

---

## 🧰 Tech Stack

| Area | Library |
|------|----------|
| Machine Learning | TensorFlow / Keras |
| Clustering | HDBSCAN |
| Dimensionality Reduction | UMAP |
| Preprocessing | scikit-learn |
| Visualization | Matplotlib |
| API Framework | Flask |

---

## 🧑‍💻 Authors

**Rishit Modi**  
💡 Developer in AI, ML  
🌐 [GitHub Profile](https://github.com/RishitModi)

---

**Mahi Desai**  
💡 Deep Learning Researcher  
🌐 [GitHub Profile](https://github.com/d-mahi14)


## ⚠️ Notes
- Requires **≥50 valid sequences** for meaningful clustering  
- Random seeds are fixed (`42`) for reproducibility  
- Mock taxa and confidences are simulated for demonstration  


⭐ *If you find this project useful, consider starring the repo to support development!* ⭐
