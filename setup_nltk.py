import nltk
import ssl
import os

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Create nltk_data directory if it doesn't exist
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

print("Downloading NLTK data...")

# Download required NLTK data
datasets = [
    'punkt',
    'stopwords', 
    'averaged_perceptron_tagger',
    'maxent_ne_chunker',
    'words',
    'wordnet',
    'omw-1.4'
]

for dataset in datasets:
    try:
        nltk.download(dataset, quiet=False)
        print(f"✓ Downloaded {dataset}")
    except Exception as e:
        print(f"✗ Failed to download {dataset}: {e}")

print("\nNLTK data download completed!")