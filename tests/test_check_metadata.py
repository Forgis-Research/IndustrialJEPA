# Check metadata structure
from huggingface_hub import hf_hub_download
import json

meta_path = hf_hub_download(
    repo_id="Forgis/factorynet-hackathon",
    filename="metadata/aursad_metadata.json",
    repo_type="dataset"
)

with open(meta_path) as f:
    metadata = json.load(f)

print(f"Type: {type(metadata)}")
print(f"Length: {len(metadata)}")

if isinstance(metadata, list):
    print(f"\nFirst item keys: {metadata[0].keys()}")
    print(f"\nFirst 3 items:")
    for m in metadata[:3]:
        print(f"  {m}")
elif isinstance(metadata, dict):
    print(f"\nKeys: {list(metadata.keys())[:5]}")
