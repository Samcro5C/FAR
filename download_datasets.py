from huggingface_hub import snapshot_download
import os
snapshot_download("guyuchao/UCF101", repo_type="dataset", local_dir="datasets/ucf101", token=os.environ['HF_HUB_TOKEN'])
