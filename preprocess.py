import os
import numpy as np
import random
import itertools
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class eDNABiodiversityPipeline:
    # Complete pipeline for AI-driven biodiversity assessment from eDNA sequences
    
    def __init__(self, n_sequences=10000, sequence_length=100, k=4):
        """
        Initialize the pipeline with key parameters.
        
        Args:
            n_sequences (int): Number of mock eDNA sequences to generate
            sequence_length (int): Length of each DNA sequence
            k (int): k-mer length for sequence vectorization
        """
        self.n_sequences = n_sequences
        self.sequence_length = sequence_length
        self.k = k
        self.sequences = None
        self.kmer_vectors = None
        self.vae_model = None
        self.embeddings = None
        self.clusters = None
        self.results = {} 
        self.plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def generate_mock_edna_sequences(self):
        """
        Generate mock eDNA sequences to simulate the output of bioinformatics preprocessing.
        
        In a real-world scenario, this would be replaced by actual ASV (Amplicon Sequence Variants)
        data from quality-filtered, denoised sequencing reads. The mock data includes:
        - Realistic base composition (A, T, G, C)
        - Variable sequence patterns to simulate different taxa
        - Some sequences with higher similarity (same taxa) and others more divergent (different taxa)
        
        Returns:
            list: Mock DNA sequences as strings
        """
        print("Generating mock eDNA sequences...")
        
        sequences = []
        bases = ['A', 'T', 'G', 'C']
    
        templates = []
        for i in range(50):  # Create 50 base templates
            template = ''.join(np.random.choice(bases, size=self.sequence_length))
            templates.append(template)
        
        # Generate sequences with controlled similarity patterns
        for i in range(self.n_sequences):
            if i < self.n_sequences * 0.7:  # 70% of sequences are variations of templates
                # Select a random template and introduce mutations
                base_template = random.choice(templates)
                sequence = list(base_template)
                
                #mutations (5-15% of positions)
                n_mutations = np.random.randint(int(0.05 * self.sequence_length), 
                                                int(0.15 * self.sequence_length))
                mutation_positions = np.random.choice(self.sequence_length, 
                                                      size=n_mutations, replace=False)
                
                for pos in mutation_positions:
                    sequence[pos] = np.random.choice(bases)
                
                sequences.append(''.join(sequence))
            else:  # 30% completely random 
                sequence = ''.join(np.random.choice(bases, size=self.sequence_length))
                sequences.append(sequence)
        
        self.sequences = sequences
        print(f"Generated {len(sequences)} mock eDNA sequences of length {self.sequence_length}")

        return sequences
    
    def sequences_to_kmers(self, sequences):
        """
        Convert DNA sequences to k-mer frequency vectors for numerical analysis.
        
        k-mers are substrings of length k that provide a way to numerically represent
        genetic sequences while preserving important sequence information. This is
        crucial for machine learning approaches.
        
        Dimensionality
        Larger k → 4ᵏ features. High-dimensional vectors can slow down downstream analyses and require more memory.
        - Coverage
        Each sequence provides (L-k+1) k-mers. If 4ᵏ ≫ (L-k+1), most features stay zero (very sparse).
        - Biological specificity
        Small k (e.g. 2-3) captures general base composition; larger k (5-6) captures more distinctive motifs but may introduce noise if count are too low.

        Returns:
            numpy.ndarray: Matrix of k-mer frequency vectors
        """
        print(f"Converting sequences to {self.k}-mer vectors...")
        
        # Generate all possible k-mers
        bases = ['A', 'T', 'G', 'C']
        from itertools import product
        all_kmers = [''.join(p) for p in product(bases, repeat=self.k)]
        kmer_to_index = {kmer: i for i, kmer in enumerate(all_kmers)}
        
        # Initialize k-mer count matrix
        kmer_matrix = np.zeros((len(sequences), len(all_kmers)))
        
        for seq_idx, sequence in enumerate(sequences):
            # Count k-mers in this sequence
            kmer_counts = Counter()
            for i in range(len(sequence) - self.k + 1):
                kmer = sequence[i:i + self.k]
                if all(base in bases for base in kmer):  
                    kmer_counts[kmer] += 1
        
            total_kmers = sum(kmer_counts.values())
            if total_kmers > 0:
                for kmer, count in kmer_counts.items():
                    if kmer in kmer_to_index:
                        kmer_matrix[seq_idx, kmer_to_index[kmer]] = count / total_kmers
        
        print(f"Created k-mer matrix of shape {kmer_matrix.shape}")
        return kmer_matrix
    
    def plot_kmer_heatmap(self, max_samples=50, figsize=(12,8), cmap='viridis'):
        if self.kmer_vectors is None:
            print("No k-mer vectors found. Run sequences_to_kmers() first.")
            return

        kmer_matrix = self.kmer_vectors
        n_seq, n_kmers = kmer_matrix.shape

        # Generate k-mer labels
        bases = ['A', 'T', 'G', 'C']
        from itertools import product
        kmer_labels = [''.join(p) for p in product(bases, repeat=self.k)]

        # Sample sequences if too many
        if n_seq > max_samples:
            idx = np.random.choice(n_seq, size=max_samples, replace=False)
        else:
            idx = np.arange(n_seq)

        sub_matrix = kmer_matrix[idx, :]
        sub_labels = [f"Seq_{i}" for i in idx]

        df = pd.DataFrame(sub_matrix, index=sub_labels, columns=kmer_labels)

        plt.figure(figsize=figsize)
        sns.heatmap(df, cmap=cmap, cbar_kws={'label': 'Frequency'})
        plt.title(f"{self.k}-mer Frequency Heatmap")
        plt.xlabel("k-mers")
        plt.ylabel("Sequences")
        plt.tight_layout()
        
        plt.savefig("kmer-heatmap.png", dpi=300)  # dpi=300 for high quality
        
        plt.show()
    
if __name__ == "__main__":
    pipeline = eDNABiodiversityPipeline()
    pipeline.generate_mock_edna_sequences()
    pipeline.kmer_vectors = pipeline.sequences_to_kmers(pipeline.sequences)
    pipeline.plot_kmer_heatmap(max_samples=20)