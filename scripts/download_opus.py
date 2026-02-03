#!/usr/bin/env python3
"""
OPUS OpenSubtitles ja-ko コーパスダウンロードスクリプト

Usage:
    python scripts/download_opus.py
"""

import os
import urllib.request
import gzip
import shutil
from pathlib import Path

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# OPUS OpenSubtitles ja-ko
OPUS_BASE_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses"
FILES = {
    "ja": f"{OPUS_BASE_URL}/ja-ko.txt.zip",
    "ko": f"{OPUS_BASE_URL}/ko-ja.txt.zip",
}

def download_file(url: str, dest: Path):
    """ファイルをダウンロード"""
    print(f"Downloading: {url}")
    print(f"  -> {dest}")
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # ダウンロード
    urllib.request.urlretrieve(url, dest)
    
    file_size = dest.stat().st_size / (1024 * 1024)  # MB
    print(f"  Downloaded: {file_size:.2f} MB")

def extract_zip(zip_path: Path, extract_dir: Path):
    """ZIPファイルを解凍"""
    import zipfile
    
    print(f"Extracting: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print(f"  Extracted to: {extract_dir}")

def main():
    print("=== OPUS OpenSubtitles ja-ko ダウンロード ===")
    print()
    
    # data/raw ディレクトリ作成
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    
    # ja-ko.txt.zip をダウンロード
    ja_ko_zip = DATA_RAW / "ja-ko.txt.zip"
    
    if ja_ko_zip.exists():
        print(f"既にダウンロード済み: {ja_ko_zip}")
    else:
        download_file(FILES["ja"], ja_ko_zip)
    
    # 解凍
    extract_dir = DATA_RAW / "extracted"
    extract_zip(ja_ko_zip, extract_dir)
    
    # ファイルを移動・リネーム
    # extracted/OpenSubtitles.ja-ko.ja → data/raw/OpenSubtitles.ja-ko.ja
    ja_file = extract_dir / "OpenSubtitles.ja-ko.ja"
    ko_file = extract_dir / "OpenSubtitles.ja-ko.ko"
    
    if ja_file.exists():
        shutil.move(str(ja_file), str(DATA_RAW / "OpenSubtitles.ja-ko.ja"))
        print(f"Moved: {ja_file.name}")
    
    if ko_file.exists():
        shutil.move(str(ko_file), str(DATA_RAW / "OpenSubtitles.ja-ko.ko"))
        print(f"Moved: {ko_file.name}")
    
    # 一時ファイル削除
    shutil.rmtree(extract_dir, ignore_errors=True)
    ja_ko_zip.unlink(missing_ok=True)
    
    print()
    print("=== ダウンロード完了 ===")
    print(f"Japanese: {DATA_RAW / 'OpenSubtitles.ja-ko.ja'}")
    print(f"Korean:   {DATA_RAW / 'OpenSubtitles.ja-ko.ko'}")
    print()
    print("次のステップ:")
    print("  python scripts/clean.py     # データクレンジング")

if __name__ == "__main__":
    main()
