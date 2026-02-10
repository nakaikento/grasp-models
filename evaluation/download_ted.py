#!/usr/bin/env python3
"""
TED Talk downloader for Korean→Japanese translation evaluation.

Downloads:
1. Audio (Korean speaker)
2. Korean subtitles (ASR reference)
3. Japanese subtitles (translation reference)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def download_ted_talk(url: str, output_dir: Path):
    """
    Download TED talk audio and subtitles (Korean + Japanese).
    
    Args:
        url: TED talk URL or YouTube URL
        output_dir: Directory to save files
    """
    audio_dir = output_dir / "audio"
    subs_dir = output_dir / "subtitles"
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    subs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading TED talk: {url}")
    
    # Extract video ID for filename
    cmd_id = [
        "yt-dlp",
        "--print", "id",
        url
    ]
    
    try:
        video_id = subprocess.check_output(cmd_id, text=True).strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to get video ID: {e}")
        return False
    
    print(f"🆔 Video ID: {video_id}")
    
    # Download audio (Korean speaker)
    print("\n🎵 Downloading audio...")
    audio_file = audio_dir / f"{video_id}.wav"
    
    cmd_audio = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "wav",
        "--audio-quality", "0",  # Best quality
        "-o", str(audio_file.with_suffix("")),  # yt-dlp adds extension
        url
    ]
    
    try:
        subprocess.run(cmd_audio, check=True)
        print(f"✅ Audio saved: {audio_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download audio: {e}")
        return False
    
    # Download Korean subtitles
    print("\n📝 Downloading Korean subtitles...")
    ko_sub = subs_dir / f"{video_id}.ko.vtt"
    
    cmd_ko = [
        "yt-dlp",
        "--write-sub",
        "--sub-lang", "ko",
        "--skip-download",
        "--convert-subs", "vtt",
        "-o", str(subs_dir / video_id),
        url
    ]
    
    try:
        subprocess.run(cmd_ko, check=True)
        print(f"✅ Korean subtitles saved: {ko_sub}")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to download Korean subtitles: {e}")
        print("    (Korean subtitles may not be available)")
    
    # Download Japanese subtitles
    print("\n📝 Downloading Japanese subtitles...")
    ja_sub = subs_dir / f"{video_id}.ja.vtt"
    
    cmd_ja = [
        "yt-dlp",
        "--write-sub",
        "--sub-lang", "ja",
        "--skip-download",
        "--convert-subs", "vtt",
        "-o", str(subs_dir / video_id),
        url
    ]
    
    try:
        subprocess.run(cmd_ja, check=True)
        print(f"✅ Japanese subtitles saved: {ja_sub}")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to download Japanese subtitles: {e}")
        print("    (Japanese subtitles may not be available)")
    
    print("\n" + "="*60)
    print("✅ Download complete!")
    print(f"📁 Files saved in: {output_dir}")
    print("="*60)
    
    return True


def check_dependencies():
    """Check if yt-dlp is installed."""
    try:
        subprocess.run(["yt-dlp", "--version"], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL,
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ yt-dlp not found. Please install:")
        print("   pip install yt-dlp")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download TED talks for Korean→Japanese evaluation"
    )
    parser.add_argument(
        "url",
        help="TED talk URL (e.g., https://www.youtube.com/watch?v=...)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Output directory (default: ./data)"
    )
    
    args = parser.parse_args()
    
    if not check_dependencies():
        sys.exit(1)
    
    success = download_ted_talk(args.url, args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
