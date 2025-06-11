import os
import re

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(REPO_ROOT, "repositories")
README_PATH = os.path.join(REPO_ROOT, "README.md")

def get_model_info():
    models = []
    datasets_set = set()
    for model_name in sorted(os.listdir(REPO_DIR)):
        model_path = os.path.join(REPO_DIR, model_name)
        if not os.path.isdir(model_path):
            continue
        scripts_dir = os.path.join(model_path, "scripts")
        if not os.path.isdir(scripts_dir):
            continue
        datasets = []
        for fname in os.listdir(scripts_dir):
            if fname.endswith(".sh"):
                dataset = fname[:-3]
                datasets.append(dataset)
                datasets_set.add(dataset)
        # URL 추출
        url = None
        orig_html = os.path.join(model_path, "original.html")
        if os.path.isfile(orig_html):
            with open(orig_html, encoding="utf-8") as f:
                url = f.read()
        models.append({
            "name": model_name,
            "datasets": set(datasets),
            "url": url or "#"
        })
    return models, sorted(datasets_set)

def make_markdown_table(models, datasets):
    header = ["Model"] + datasets
    lines = []
    # 헤더
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["-"*len(h) for h in header]) + "|")
    # 각 모델
    for m in models:
        url = m["url"]
        name = m["name"]
        row = [f'<a href="{url}">{name}</a>']
        for d in datasets:
            row.append(":white_check_mark:" if d in m["datasets"] else ":x:")
        lines.append("|" + " | ".join(row) + "|")
    return "\n".join(lines)

def update_readme(table_md):
    with open(README_PATH, encoding="utf-8") as f:
        content = f.read()
    new_content = re.sub(
        r"(<!--  Registered Models begin-->)(.*?)(<!--  Registered Models end-->)",
        r"\1\n# Registered Models\n\n" + table_md + r"\n\n\3",
        content,
        flags=re.DOTALL
    )
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)

if __name__ == "__main__":
    models, datasets = get_model_info()
    table_md = make_markdown_table(models, datasets)
    update_readme(table_md)
    print("README.md Registered Models section updated.")