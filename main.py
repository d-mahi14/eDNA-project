from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import analysis # Your provided python script, renamed to analysis.py
import tempfile
import os
import shutil
import base64
import uuid

app = Flask(__name__, static_folder='static', template_folder='templates')

def encode_image_to_base64(path):
    """Encodes an image file to a base64 string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError:
        return None

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sequence_route():
    """The main analysis endpoint."""
    data = request.get_json()
    if not data or 'sequence' not in data:
        return jsonify({'error': 'No sequence data provided'}), 400

    sequence_text = data['sequence']
    sequences = [seq.strip() for seq in sequence_text.splitlines() if seq.strip()]
    
    if not sequences or len(sequences) < 10:
        return jsonify({'error': 'Please provide at least 10 valid DNA sequences.'}), 400

    # Create a unique temporary directory for this request
    temp_dir = tempfile.mkdtemp()
    
    try:
        # --- STEP 1: K-mer counting ---
        data_pipeline = analysis.eDNAPipeline_Data()
        kmer_vectors = data_pipeline.sequences_to_kmers(sequences)

        # --- STEP 2: VAE Training & Embedding ---
        embeddings = analysis.train_vae_and_get_embeddings(kmer_vectors)
        
        # Define paths within the temp directory
        embeddings_path = os.path.join(temp_dir, 'embeddings.npy')
        k_dist_path = os.path.join(temp_dir, 'k_distance.png')
        scatter_path = os.path.join(temp_dir, 'latent_clusters.png')
        
        np.save(embeddings_path, embeddings)

        # --- STEP 3: Clustering and Analysis ---
        bio_pipeline = analysis.eDNABiodiversityPipeline()
        
        # Load embeddings (from temp dir, though we already have it in memory)
        loaded_embeddings = bio_pipeline.load_embeddings(embeddings_path)
        
        # Generate plots
        bio_pipeline.plot_k_distance(loaded_embeddings, save_path=k_dist_path)
        
        # Run DBSCAN
        # Tune eps and min_samples here if needed
        labels = bio_pipeline.run_dbscan(loaded_embeddings, eps=1.5, min_samples=5)
        
        # Generate scatter plot
        bio_pipeline.save_cluster_scatter(loaded_embeddings, labels, save_path=scatter_path)
        
        # Get biodiversity metrics
        summary_data = bio_pipeline.analyze_biodiversity(labels)
        
        # --- PREPARE RESPONSE ---
        # Encode plots to base64
        k_dist_b64 = encode_image_to_base64(k_dist_path)
        scatter_b64 = encode_image_to_base64(scatter_path)
        
        final_response = {
            'summary': summary_data,
            'plots': {
                'pca_scatter': scatter_b64,
                'k_distance': k_dist_b64
            }
        }
        
        return jsonify(final_response)

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred: {e}")
        return jsonify({'error': 'An internal error occurred during analysis.'}), 500

    finally:
        # Clean up the temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

