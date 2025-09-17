

"""
Integration: This module is indirectly invoked by simulators/agents through adapter or tooling calls.
Rationale: Operational tooling invoked by agents/simulators when required.
"""
from huggingface_hub import HfApi

def find_motion_datasets():
    """
    Searches the Hugging Face Hub for datasets tagged with 'motion'.
    """
    api = HfApi()
    # Broaden the search to be more inclusive
    datasets = api.list_datasets(
        search="humanoid",
        tags=["motion"],
        sort="likes",
        direction=-1
    )

    if not datasets:
        print("No datasets found with the tags 'motion' and 'humanoid' in the search.")
        print("Checking for 'HumanML3D' as a fallback...")
        datasets = api.list_datasets(search="HumanML3D")

    if datasets:
        print("Found the following potential motion datasets:")
        for i, dataset in enumerate(datasets):
            print(f"{i+1}. ID: {dataset.id}, Likes: {dataset.likes}")
            if i >= 10: # Print top 10
                break
    else:
        print("Could not find any suitable motion datasets on the Hub.")

if __name__ == "__main__":
    find_motion_datasets()
