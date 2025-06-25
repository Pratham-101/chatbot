import os
from google.cloud import storage

# --- GCP Service Account Key for Render ---
if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
    with open("/tmp/gcp_key.json", "w") as f:
        f.write(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp_key.json"
# --- End GCP Service Account Key for Render ---

BUCKET_NAME = "mutualfundpro-vectorstore"
VECTOR_STORE_DIR = "vector_store"

def download_blob_dir(bucket_name, source_blob_prefix, destination_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_blob_prefix)
    for blob in blobs:
        rel_path = os.path.relpath(blob.name, source_blob_prefix)
        dest_path = os.path.join(destination_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        blob.download_to_filename(dest_path)
        print(f"Downloaded {blob.name} to {dest_path}")

if __name__ == "__main__":
    if not os.path.exists(VECTOR_STORE_DIR):
        print("Vector store not found locally. Downloading from GCS...")
        download_blob_dir(BUCKET_NAME, "vector_store", VECTOR_STORE_DIR)
    else:
        print("Vector store already exists locally. Skipping download.") 