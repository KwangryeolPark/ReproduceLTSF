import os
import argparse
import yaml

parser = argparse.ArgumentParser(description="Clone Git repositories and save metadata.")
parser.add_argument("--repo_dir", type=str, help="Git repository URL ending with `.git`", required=True)
parser.add_argument("--conference", type=str, help="Conference name", required=True)
parser.add_argument("--year", type=int, help="Year of the conference", required=True)
parser.add_argument("--model_name", type=str, help="Model name", default=None)
args = parser.parse_args()

clone_repo_dir = args.repo_dir
repo_dir = clone_repo_dir.replace('/tree/main', '').replace('/tree/master', '').replace('.git', '')
conference = args.conference
year = args.year

if conference == 'NIPS':
    conference = 'NeurIPS'

repo_model_name = repo_dir.split('/')[-1]
if args.model_name == None:
    model_name = repo_model_name
else:
    model_name = args.model_name

if not os.path.exists('repositories'):
    os.makedirs('repositories')

# Model Path
model_path = os.path.join('repositories', repo_model_name) if args.model_name == None else os.path.join('repositories', model_name)
   
# Clone the repository if it does not exist
if not os.path.exists(model_path):
    if args.model_name == None:
        os.system(f"cd repositories && git clone {clone_repo_dir}")
    else:
        os.system(f"cd repositories && git clone {clone_repo_dir} && mv {repo_model_name} {model_name}")
        
    
# Remove files and folders that are not needed
remove_list = ['.git', '.gitignore', 'README.md', 'requirements.txt', 'venv', 'dataset', 'logs', 'checkpoints', 'results', 'test_results']
for item in remove_list:
    item_path = os.path.join(model_path, item)
    if os.path.exists(item_path):
        if os.path.isdir(item_path):
            os.system(f"rm -rf {item_path}")
        else:
            os.remove(item_path)
 
# Save metadata
metadata = {
    "repo_dir": repo_dir,
    "conference": conference,
    "year": year
}
metadata_file = os.path.join(model_path, 'metadata.yaml')


if not os.path.exists(metadata_file):
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, indent=4)

# Symlink the model path
links = ['venv', 'dataset', 'requirements.txt']

for link in links:
    link_path = os.path.join(model_path, link)
    if not os.path.exists(link_path):
        os.system(f"cd {model_path} && ln -s ../../{link} {link}")
    else:
        # Remove existing symlink if it exists
        os.system(f"cd {model_path} && rm -rf {link} && ln -s ../../{link} {link}")

# Create necessary directories
necessary_dirs = ['logs', 'checkpoints', 'results', 'test_results']
for d in necessary_dirs:
    dir_path = os.path.join(model_path, d)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Symlink the directories
links = ['results', 'test_results']
for link in links:
    if not os.path.exists(link):
        os.makedirs(link)
    link_path = os.path.join(link, model_name)
    if not os.path.exists(link_path):
        os.system(f"cd {link} && ln -s ../{model_path}/{link} {model_name}")
    
        
# Modify README file
readme_path = os.path.join(model_path, 'README.md')
with open(readme_path, 'w') as f:
    f.write(f"# Important\n")
    f.write(f"* This repository is the clone version of {repo_dir}\n")


model_name = model_name.split('/')[-1]
os.system(f'python copy_notebooks.py --repo_dir {model_name}')