from huggingface_hub import snapshot_download, hf_hub_download
import os

snapshot_download(
    repo_id="guyuchao/FAR_Models",
    repo_type="model",
    local_dir="experiments/pretrained_models/FAR_Models",
    token=os.environ['HF_HUB_TOKEN']
)