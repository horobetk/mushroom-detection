import os

# Конфигурация: что игнорируем, чтобы слепок не весил 100Мб
EXCLUDE_DIRS = {'.git', '__pycache__', '.venv', 'venv', 'datasets', 'runs', 'external', 'processed', '.idea', '.vscode'}
EXCLUDE_FILES = {'project_snapshot.txt', 'make_snapshot.py', 'yolov8x.pt', 'yolov8n.pt'}
EXTENSIONS = {'.py', '.md', '.yaml', '.txt', '.json', '.bat'}

def generate_snapshot(root_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"PROJECT SNAPSHOT - {os.path.basename(os.path.abspath(root_dir))}\n")
        f.write("="*50 + "\n\n")
        
        f.write("DIRECTORY STRUCTURE:\n")
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            level = root.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                if file not in EXCLUDE_FILES:
                    f.write(f"{sub_indent}{file}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write("FILE CONTENTS:\n")
        f.write("="*50 + "\n\n")
        
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                if any(file.endswith(ext) for ext in EXTENSIONS) and file not in EXCLUDE_FILES:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, root_dir)
                    f.write(f"--- FILE: {relative_path} ---\n")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as content_f:
                            f.write(content_f.read())
                    except Exception as e:
                        f.write(f"[Error reading file: {e}]")
                    f.write("\n\n")

if __name__ == "__main__":
    generate_snapshot('.', 'project_snapshot.txt')
    print("Слепок проекта готов: project_snapshot.txt")