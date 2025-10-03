# -- coding: utf-8 --
"""
eDNA Biodiversity Analysis Pipeline
Version 4.7 - Refined taxonomic classification and summary reporting.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter
from itertools import product
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import warnings

try:
    import hdbscan
    import umap
except ImportError:
    hdbscan = None
    umap = None

warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

def generate_mock_edna_sequences(n_sequences=20000, sequence_length=200):
    """Generates a mock dataset of eDNA sequences."""
    print(f"--- Generating {n_sequences} mock eDNA sequences... ---")
    sequences = []
    bases = ['A', 'T', 'G', 'C']
    templates = [''.join(np.random.choice(bases, size=sequence_length)) for _ in range(10)]

    for i in range(n_sequences):
        if i < n_sequences * 0.7:
            base_template = random.choice(templates)
            sequence = list(base_template)
            n_mutations = np.random.randint(int(0.15 * sequence_length), int(0.25 * sequence_length))
            mutation_positions = np.random.choice(sequence_length, size=n_mutations, replace=False)
            for pos in mutation_positions:
                sequence[pos] = np.random.choice(list(set(bases) - {sequence[pos]}))
            sequences.append(''.join(sequence))
        else:
            sequences.append(''.join(np.random.choice(bases, size=sequence_length)))
    
    print(f"Generated {len(sequences)} mock sequences.")
    return sequences

def sequences_to_kmers(sequences, k=4):
    """Converts raw DNA sequences into normalized k-mer frequency vectors."""
    print(f"--- Converting sequences to {k}-mer vectors... ---")
    bases = ['A', 'T', 'G', 'C']
    all_kmers = [''.join(p) for p in product(bases, repeat=k)]
    kmer_to_index = {kmer: i for i, kmer in enumerate(all_kmers)}
    kmer_matrix = np.zeros((len(sequences), len(all_kmers)))

    for seq_idx, sequence in enumerate(sequences):
        clean_sequence = ''.join(filter(lambda base: base in 'ATGC', sequence.upper()))
        if not clean_sequence: continue
        
        kmer_counts = Counter([clean_sequence[i:i + k] for i in range(len(clean_sequence) - k + 1)])
        total_kmers = sum(kmer_counts.values())

        if total_kmers > 0:
            for kmer, count in kmer_counts.items():
                if kmer in kmer_to_index:
                    kmer_matrix[seq_idx, kmer_to_index[kmer]] = count / total_kmers
    
    print(f"Created k-mer matrix of shape {kmer_matrix.shape}")
    return kmer_matrix

class VAE(keras.Model):
    def __init__(self, original_dim, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential([
            keras.Input(shape=(original_dim,)),
            layers.Dense(256, activation="relu"), layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            layers.Dense(latent_dim * 2),
        ])
        self.decoder = keras.Sequential([
            keras.Input(shape=(latent_dim,)),
            layers.Dense(128, activation="relu"), layers.Dropout(0.2),
            layers.Dense(256, activation="relu"),
            layers.Dense(original_dim, activation="sigmoid"),
        ])
    def reparameterize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch, self.latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    def call(self, inputs):
        z_mean, z_log_var = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        self.add_loss(kl_loss)
        return reconstruction
    def get_embeddings(self, inputs):
        z_mean, _ = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        return z_mean

def train_vae_and_get_embeddings(kmer_matrix, epochs=30, batch_size=128):
    print("--- Starting VAE Training ---")
    scaler = StandardScaler()
    kmer_scaled = scaler.fit_transform(kmer_matrix)
    vae = VAE(original_dim=kmer_scaled.shape[1])
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(kmer_scaled, kmer_scaled, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
    embeddings = vae.get_embeddings(kmer_scaled)
    print("--- VAE Training and Embedding Extraction Complete ---")
    return embeddings.numpy()

def run_hdbscan(embeddings, min_cluster_size=10, min_samples=None):
    print(f"--- Running HDBSCAN (min_cluster_size={min_cluster_size}) ---")
    if hdbscan is None:
        raise ImportError("HDBSCAN library not found. Please install it via 'pip install hdbscan'.")
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
    labels = clusterer.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"HDBSCAN found {n_clusters} clusters and {n_noise} noise points.")
    return labels

def get_consensus_sequence(sequences, cluster_labels):
    print("--- Generating Consensus Sequences for Clusters ---")
    consensus_map = {}
    unique_clusters = sorted(set(cluster_labels))
    sequences_np = np.array(sequences)
    for cluster_id in unique_clusters:
        if cluster_id == -1: continue
        indices = np.where(cluster_labels == cluster_id)[0]
        cluster_sequences = sequences_np[indices]
        if len(cluster_sequences) == 0: continue
        seq_len = len(cluster_sequences[0])
        consensus_seq = []
        for i in range(seq_len):
            base_counts = Counter(seq[i] for seq in cluster_sequences)
            if not base_counts:
                consensus_seq.append('N'); continue
            max_count = max(base_counts.values())
            most_common = [base for base, count in base_counts.items() if count == max_count]
            consensus_seq.append(random.choice(most_common) if most_common else 'N')
        consensus_map[cluster_id] = "".join(consensus_seq)
    print("Generated consensus sequences for all identified clusters.")
    return consensus_map

def assign_partial_taxonomy_to_noise(embeddings, sequences, labels, taxa_details):
    print("--- Analyzing Noise Points for Partial Taxonomy... ---")
    noise_indices = np.where(labels == -1)[0]
    if len(noise_indices) == 0:
        print("No noise points to analyze."); return []
    cluster_id_to_taxon = {t['cluster_id']: t for t in taxa_details}
    unique_labels = sorted([l for l in np.unique(labels) if l != -1])
    if not unique_labels:
        print("No valid clusters found to assign noise to. Skipping."); return []
    cluster_centroids = np.array([embeddings[labels == i].mean(axis=0) for i in unique_labels])
    noise_embeddings = embeddings[noise_indices]
    distances = cdist(noise_embeddings, cluster_centroids, metric='euclidean')
    closest_cluster_indices = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)
    sequences_np = np.array(sequences)
    new_taxa_list = []
    for i, noise_idx in enumerate(noise_indices):
        best_match_cluster_id = unique_labels[closest_cluster_indices[i]]
        closest_taxon_info = cluster_id_to_taxon.get(best_match_cluster_id)
        if not closest_taxon_info: continue
        novel_taxon = {
            'noise_point_id': int(noise_idx),
            'closest_known_cluster_id': int(best_match_cluster_id),
            'closest_taxon_name': closest_taxon_info['taxon_name'],
            'distance': round(min_distances[i], 4),
            'sequence': sequences_np[noise_idx]
        }
        new_taxa_list.append(novel_taxon)
    print(f"Processed {len(noise_indices)} noise points.")
    return sorted(new_taxa_list, key=lambda x: x['distance'])

def save_cluster_scatter(embeddings, labels, save_path):
    print("--- Generating UMAP Cluster Visualization ---")
    if umap is None:
        raise ImportError("UMAP library not found. Please install it via 'pip install umap-learn'.")
    reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=15)
    X_umap = reducer.fit_transform(embeddings)
    unique_labels = sorted(set(labels))
    colors = plt.cm.get_cmap('viridis', len(unique_labels))
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        label_name = 'Noise' if lab == -1 else f'Cluster {lab}'
        color = 'gray' if lab == -1 else colors(i)
        size = 10 if lab == -1 else 25
        alpha = 0.4 if lab == -1 else 0.9
        marker = 'x' if lab == -1 else 'o'
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], s=size, alpha=alpha, label=label_name, color=color, marker=marker)
    ax.set_title('HDBSCAN Clusters (UMAP Projection)', fontsize=16, color='white')
    ax.set_xlabel('UMAP Dimension 1', fontsize=12, color='white')
    ax.set_ylabel('UMAP Dimension 2', fontsize=12, color='white')
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')
    if len(unique_labels) < 15:
        legend = ax.legend(markerscale=2)
        plt.setp(legend.get_texts(), color='white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Cluster scatter plot saved to {save_path}")

def analyze_biodiversity(labels, embeddings, sequences):
    """Calculates biodiversity metrics and generates a comprehensive summary."""
    print("--- Analyzing Biodiversity ---")
    cluster_counts = Counter(labels)
    
    taxa_details = []
    mock_taxa_names = ["Vibrio", "Bacillus", "Pseudoalteromonas", "Halomonas", "Marinobacter", "Alteromonas", "Shewanella"]
    novel_taxa_count = 0

    for cluster_id in sorted([c for c in cluster_counts if c != -1]):
        count = cluster_counts[cluster_id]
        is_novel = random.random() > 0.8  # Simulate ~20% of clusters being novel
        
        if is_novel:
            novel_taxa_count += 1
            taxa_details.append({
                'cluster_id': int(cluster_id), 'abundance': int(count), 'status': 'Novel',
                'confidence': 0.0, 'taxon_name': f"Unclassified OTU {cluster_id}",
                'taxonomic_level': 'Unknown'
            })
        else:
            taxa_details.append({
                'cluster_id': int(cluster_id), 'abundance': int(count), 'status': 'Identified',
                'confidence': round(random.uniform(0.85, 0.99), 3),
                'taxon_name': f"{random.choice(mock_taxa_names)} sp. C{cluster_id}",
                'taxonomic_level': 'Genus'
            })

    # Noise points are handled separately and not counted in biodiversity metrics
    noise_point_details = assign_partial_taxonomy_to_noise(embeddings, sequences, labels, taxa_details)

    # Calculate metrics based only on clustered (non-noise) data
    abundances = [t['abundance'] for t in taxa_details]
    total_in_clusters = sum(abundances)
    richness = len(taxa_details)
    shannon = -np.sum([(p/total_in_clusters) * np.log(p/total_in_clusters) for p in abundances if p > 0]) if total_in_clusters > 0 else 0
    simpson = 1 - sum((p/total_in_clusters)**2 for p in abundances) if total_in_clusters > 0 else 0
    pielou = shannon / np.log(richness) if richness > 1 else 0

    summary_data = {
        'biodiversity_metrics': {
            'species_richness': int(richness),
            'shannon_diversity_index': round(shannon, 4),
            'simpson_diversity_index': round(simpson, 4),
            'pielou_evenness_index': round(pielou, 4),
        },
        'taxonomic_summary': {
            'identified_taxa': int(richness - novel_taxa_count),
            'novel_taxa': int(novel_taxa_count),
            'noise_points': int(cluster_counts.get(-1, 0)),
        },
        'taxa_details': sorted(taxa_details, key=lambda x: x['abundance'], reverse=True),
        'noise_point_details': noise_point_details
    }
    
    print("Biodiversity analysis complete.")
    return summary_data

