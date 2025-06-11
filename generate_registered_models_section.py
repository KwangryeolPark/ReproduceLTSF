import os
import re
import yaml
import pandas as pd

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
        # Meta 정보 추출
        meta_file = os.path.join(model_path, "metadata.yaml")
        metadata = {}
        if os.path.isfile(meta_file):
            with open(meta_file, encoding="utf-8") as f:
                metadata = yaml.safe_load(f)
        conference = metadata.get("conference", "Unknown")
        url = metadata.get("repo_dir", "#")
        year = metadata.get("year", "Unknown")
        
        models.append({
            "name": model_name,
            "datasets": set(datasets),
            "url": url,
            "conference": conference,
            "year": year
        })

    # Sort models by year and name
    models.sort(key=lambda x: (x["year"], x["name"]))

    # Make the year a string for consistent formatting
    for model in models:
        model["year"] = str(model["year"])

    return models, sorted(datasets_set)

def make_markdown_table(models, datasets):
    header = ["Model"] + ["Conference"] + ["Year"] + datasets
    lines = []
    # 모델 개수 표기
    lines.append(f"**# of models: {len(models)}**\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["-"*len(h) for h in header]) + "|")
    for m in models:
        url = m["url"]
        name = m["name"]
        row = [f'<a href="{url}">{name}</a>']
        row.append(m["conference"])
        row.append(m["year"])
        for d in datasets:
            row.append(":white_check_mark:" if d in m["datasets"] else ":x:")
        lines.append("|" + " | ".join(row) + "|")
    return "\n".join(lines)

def make_dataframe(models, datasets):
    rows = []
    for m in models:
        row = {
            "Model": m["name"],
            "Conference": m["conference"],
            "Year": m["year"],
        }
        for d in datasets:
            row[d] = "O" if d in m["datasets"] else "X"
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

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

    # Summary
    df = make_dataframe(models, datasets)
    print(f"\n[Registered Models Summary] ({len(models)} Models)")
    print(df)