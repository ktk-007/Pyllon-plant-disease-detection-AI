from huggingface_hub import HfApi, create_repo
import os
import sys

token = os.getenv("HF_TOKEN")
api = HfApi(token=token)

try:
    # Get user info
    user_info = api.whoami()
    username = user_info["name"]
    repo_name = f"{username}/pyllon-models"
    
    print(f"Creating repository: {repo_name}")
    create_repo(repo_id=repo_name, token=token, exist_ok=True, repo_type="model", private=False)
    
    print("Starting upload of 3GB models. Please wait...")
    api.upload_folder(
        folder_path="models",
        repo_id=repo_name,
        repo_type="model",
        token=token
    )
    print(f"SUCCESS: Models successfully uploaded to {repo_name}")
    
    # Update app.py
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace placeholder
    new_content = content.replace('repo_id="YOUR_HF_USERNAME/pyllon-models"', f'repo_id="{repo_name}"')
    
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("app.py updated with the correct repo_id!")
    
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
