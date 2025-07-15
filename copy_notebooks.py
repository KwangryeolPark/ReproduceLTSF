import os
import shutil
import argparse

parser = argparse.ArgumentParser(description="Copy Jupyter notebooks from repositories to a specified directory.")
parser.add_argument("--repo_dir", type=str, help="Path to the directory containing repositories", required=True)
parser.add_argument("--overwrite", action='store_true', help="Overwrite existing notebooks in the destination directory")
args = parser.parse_args()

def copy_notebooks(repo_dir, overwrite):
    original_notebook_path = 'notebooks'

    if not os.path.exists(original_notebook_path):
        raise FileNotFoundError(f"The original path '{original_notebook_path}' does not exist.")

    notebook_list = [notebook for notebook in os.listdir(original_notebook_path) if notebook.endswith('.ipynb')]

    destination_path = os.path.join('repositories', repo_dir, 'notebooks')
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    for notebook in notebook_list:
        source_file = os.path.join(original_notebook_path, notebook)
        destination_file = os.path.join(destination_path, notebook)

        if os.path.exists(destination_file) and not overwrite:
            print(f"Skipping {notebook}, already exists in the destination directory.")
        else:
            shutil.copy2(source_file, destination_file)
            print(f"Copied {notebook} to {destination_path}.")
            
if __name__ == "__main__":
    if args.repo_dir == 'all':
        repo_dirs = [d for d in os.listdir('repositories') if os.path.isdir(os.path.join('repositories', d))]
        for repo_dir in repo_dirs:
            copy_notebooks(repo_dir, args.overwrite)
    else:
        copy_notebooks(args.repo_dir, args.overwrite)
        print(f"Copied notebooks to {args.repo_dir}.")
    print("Notebook copying completed.")