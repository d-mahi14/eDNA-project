import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ===================================================================
# 1. VAE Model Definition 🤖
# ===================================================================
class VAE(keras.Model):
    """A Variational Autoencoder to learn embeddings from k-mer vectors."""
    def __init__(self, original_dim, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential([
            keras.Input(shape=(original_dim,)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            layers.Dense(latent_dim + latent_dim),
        ])
        self.decoder = keras.Sequential([
            keras.Input(shape=(latent_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(256, activation="relu"),
            layers.Dense(original_dim, activation="sigmoid"),
        ])

    def reparameterize(self, z_mean, z_log_var):
        """The reparameterization trick."""
        batch = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch, self.latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        """The forward pass of the model."""
        z_mean, z_log_var = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        # Calculate and add the KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        self.add_loss(kl_loss)
        return reconstruction

    def get_embeddings(self, inputs):
        """Extracts the latent space embeddings (z_mean) from the encoder."""
        z_mean, _ = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        return z_mean

# ===================================================================
# 2. VAE Training Function ⚙️
# ===================================================================
def train_vae_and_get_embeddings(kmer_matrix, epochs=30, batch_size=128):
    """Trains the VAE and returns the learned sequence embeddings."""
    print("\n--- Starting VAE Training ---")
    scaler = StandardScaler()
    kmer_scaled = scaler.fit_transform(kmer_matrix)
    
    vae = VAE(original_dim=kmer_scaled.shape[1])
    vae.compile(optimizer=keras.optimizers.Adam())
    
    print(f"Training VAE for {epochs} epochs...")
    vae.fit(kmer_scaled, kmer_scaled, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    
    embeddings = vae.get_embeddings(kmer_scaled)
    print("--- VAE Training and Embedding Extraction Complete ---")
    return embeddings.numpy()

# ===================================================================
# 3. Execution Block 🚀
# ===================================================================
# This assumes an object 'pipeline_data' with the attribute 'kmer_vectors'
# was created and populated by running the first cell.
if 'pipeline_data' in locals() and hasattr(pipeline_data, 'kmer_vectors') and pipeline_data.kmer_vectors is not None:
    # Run the VAE training on the k-mer data from Cell 1
    sequence_embeddings = train_vae_and_get_embeddings(pipeline_data.kmer_vectors)
    
    # Save the embeddings to a file for your partner
    print("\nSaving deliverables...")
    np.save('embeddings.npy', sequence_embeddings)
    print("✅ Successfully saved 'embeddings.npy'. Your partner can now use this file for clustering.")
    
else:
    print("❌ Error: Could not find 'pipeline_data' or its 'kmer_vectors'.")
    print("Please make sure you have successfully run the first cell to generate the data.")