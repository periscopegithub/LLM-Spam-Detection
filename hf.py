from huggingface_hub import HfApi, HfFolder

api = HfApi()
repo_id = "periscopehf/capstone-email-classifier"

# Create a new repo if it doesn't exist
api.create_repo(repo_id=repo_id)

# Upload files
api.upload_folder(
    folder_path=r".\experiments\checkpoint-26835",
    path_in_repo=".",
    repo_id=repo_id,
)
