#!/usr/bin/env python3
"""
Convert VTT subtitles to plain text.
"""

import re
import argparse
from pathlib import Path


def vtt_to_text(vtt_path: Path, output_path: Path = None):
    """
    Convert VTT subtitle file to plain text.
    
    Args:
        vtt_path: Path to .vtt file
        output_path: Path to output .txt file (optional)
    
    Returns:
        str: Plain text content
    """
    if not vtt_path.exists():
        raise FileNotFoundError(f"VTT file not found: {vtt_path}")
    
    with open(vtt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    text_lines = []
    skip_next = False
    
    for line in lines:
        line = line.strip()
        
        # Skip WEBVTT header
        if line.startswith('WEBVTT'):
            continue
        
        # Skip timestamp lines (e.g., "00:00:01.234 --> 00:00:03.456")
        if '-->' in line:
            skip_next = False
            continue
        
        # Skip cue identifiers (numbers or IDs)
        if line.isdigit():
            continue
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip styling tags (e.g., <v Speaker>)
        if line.startswith('<') and line.endswith('>'):
            continue
        
        # Remove HTML tags
        clean_line = re.sub(r'<[^>]+>', '', line)
        
        if clean_line:
            text_lines.append(clean_line)
    
    # Join with spaces (for Korean/Japanese text)
    full_text = ' '.join(text_lines)
    
    # Remove multiple spaces
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    
    # Save to file if output_path is provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"✅ Saved: {output_path}")
    
    return full_text


def main():
    parser = argparse.ArgumentParser(description="Convert VTT to plain text")
    parser.add_argument("vtt_file", type=Path, help="Input VTT file")
    parser.add_argument("-o", "--output", type=Path, help="Output text file")
    
    args = parser.parse_args()
    
    text = vtt_to_text(args.vtt_file, args.output)
    
    if not args.output:
        print(text)


if __name__ == "__main__":
    main()
