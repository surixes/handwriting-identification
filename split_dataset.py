import os
import shutil
import subprocess
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ARCHIVE_PATH = r"C:/main_work/uniq/writer-identification/archive.rar"
DST_ROOT     = "CERUG-RU"
DIR     = "."
PASSWORD     = os.getenv('ARCHIVE_PASSWORD') or ""

# путь к WinRAR или 7z
UNPACKER = r"D:/ZIP/WinRAR.exe"  

def extract_archive(src=ARCHIVE_PATH, dst=DIR, pwd=PASSWORD):
    """Распаковывает архив в dst/ и возвращает путь к вложенной папке img/"""
    # если папка уже распакована, пропускаем
    existing = os.path.join(dst, '20200923_Dataset_Words_Public')
    if os.path.isdir(existing):
        print(f"Папка распакована ранее, пропускаем распаковку: {existing}")
        data_folder = os.path.join(existing, 'img')
        if not os.path.isdir(data_folder):
            raise RuntimeError(f"Ожидали подкаталог img/ в {existing}, но не нашли: {data_folder}")
        return data_folder

    os.makedirs(dst, exist_ok=True)

    # 1) Распаковываем
    cmd = [UNPACKER, "x", src, f"-o{dst}", "-y"]
    if pwd:
        cmd.append(f"-p{pwd}")
    print("Запускаем распаковку:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Unpack error:", res.stderr)
        raise RuntimeError("Не удалось распаковать архив")
    print("Архив распакован в", dst)

    # 2) Ищем подпапку вида *_Dataset_Words_Public
    subs = [d for d in os.listdir(dst)
            if os.path.isdir(os.path.join(dst, d)) 
               and d.endswith("_Dataset_Words_Public")]
    if len(subs) != 1:
        raise RuntimeError(f"Ожидали одну папку *_Dataset_Words_Public, нашли: {subs}")
    data_folder = os.path.join(dst, subs[0], "img")
    if not os.path.isdir(data_folder):
        raise RuntimeError(f"Внутри {subs[0]} нет подкаталога img/: {data_folder}")
    print("Источник изображений:", data_folder)
    return data_folder


def split_train_test(csv_path, src_root, train_dst, test_dst):
    """Берёт все файлы из src_root и раскладывает по train_dst/test_dst"""
    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(test_dst,  exist_ok=True)

    df = pd.read_csv(csv_path, dtype=str)

    # Индексируем по basename (без расширения)
    file_index = {}
    for fn in os.listdir(src_root):
        name, _ = os.path.splitext(fn)
        file_index.setdefault(name, []).append(os.path.join(src_root, fn))

    # Копируем
    for _, row in df.iterrows():
        img_id = row['id']
        stage  = row['stage'].lower()
        dst    = train_dst if stage in ('train','val') else test_dst

        paths = file_index.get(img_id)
        if not paths:
            print(f"id={img_id} not found in {src_root}")
            continue
        for p in paths:
            shutil.copy2(p, dst)

    print(f"Split done: train({len(os.listdir(train_dst))}), test({len(os.listdir(test_dst))})")
