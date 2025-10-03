from flask import Flask, request, jsonify, render_template
import analysis
import tempfile
import os
import shutil
import base64
import re

app = Flask(__name__)

def encode_image_to_base64(path):
    """Encodes an image file to a base64 string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError:
        print(f"Error reading image file: {path}")
        return None

def parse_fasta(sequence_text):
    """Parses FASTA formatted text into a list of sequences, ignoring headers."""
    text_no_headers = re.sub(r'>.*\n?', '', sequence_text)
    sequences = [seq.strip().upper() for seq in text_no_headers.splitlines() if seq.strip()]
    return sequences

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_mock_data', methods=['GET'])
def generate_mock_data_route():
    try:
        sequences = analysis.generate_mock_edna_sequences(n_sequences=1000)
        sequences_string = "".join(f">sample_sequence_{i+1}\n{seq}\n" for i, seq in enumerate(sequences))
        return jsonify({'sequences': sequences_string})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_sequence_route():
    data = request.get_json()
    if not data or 'sequence' not in data:
        return jsonify({'error': 'No sequence data provided'}), 400

    sequences = parse_fasta(data['sequence'])
    if not sequences or len(sequences) < 50:
        return jsonify({'error': 'Please provide at least 50 valid DNA sequences for meaningful clustering.'}), 400

    temp_dir = tempfile.mkdtemp()
    
    try:
        # --- Full Pipeline Execution via analysis.py ---
        kmer_vectors = analysis.sequences_to_kmers(sequences)
        embeddings = analysis.train_vae_and_get_embeddings(kmer_vectors)
        labels = analysis.run_hdbscan(embeddings, min_cluster_size=10)
        
        # Get full summary from the analysis module
        summary_data = analysis.analyze_biodiversity(labels, embeddings, sequences)
        
        # Generate and encode the plot
        scatter_path = os.path.join(temp_dir, 'latent_clusters.png')
        analysis.save_cluster_scatter(embeddings, labels, scatter_path)
        scatter_b64 = encode_image_to_base64(scatter_path)
        if not scatter_b64:
            return jsonify({'error': 'Failed to generate plot image.'}), 500

        # --- Prepare Final Response ---
        final_response = {
            'summary': summary_data,
            'plots': {'cluster_plot': scatter_b64}
        }
        
        return jsonify(final_response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

