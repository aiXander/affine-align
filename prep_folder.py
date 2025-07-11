#!/usr/bin/env python3
"""
Image preprocessing script for training pairs.
Converts ***_D2_aligned.jpg (start) and ***_D2.jpg (end) images to indexed format
with resizing and prompt text files.
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import re


def resize_image_proportional(image, max_size=1536):
    """
    Resize image proportionally so that max(width, height) = max_size
    without stretching/distorting the image.
    """
    width, height = image.size
    
    # Calculate the scaling factor
    scale_factor = max_size / max(width, height)
    
    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize using high-quality resampling
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image


def extract_base_name(filename):
    """
    Extract the base name from filenames like 'uuid_D2_aligned.jpg' or 'uuid_D2.jpg'
    Returns the 'uuid' part.
    """
    if filename.endswith('_D2_aligned.jpg'):
        return filename[:-15]  # Remove '_D2_aligned.jpg'
    elif filename.endswith('_D2.jpg'):
        return filename[:-7]   # Remove '_D2.jpg'
    else:
        return None


def find_image_pairs(input_folder):
    """
    Find all matching pairs of _D2_aligned.jpg and _D2.jpg files.
    Returns a list of tuples: (base_name, aligned_path, d2_path)
    """
    input_path = Path(input_folder)
    
    # Find all _D2_aligned.jpg files
    aligned_files = list(input_path.glob('*_D2_aligned.jpg'))
    
    pairs = []
    
    for aligned_file in aligned_files:
        base_name = extract_base_name(aligned_file.name)
        if base_name:
            # Look for corresponding _D2.jpg file
            d2_file = input_path / f"{base_name}_D2.jpg"
            if d2_file.exists():
                pairs.append((base_name, aligned_file, d2_file))
            else:
                print(f"Warning: No matching _D2.jpg found for {aligned_file.name}")
    
    return pairs


def process_images(input_folder, output_folder, max_size=1536):
    """
    Process all image pairs in the input folder and save them to output folder.
    """
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image pairs
    pairs = find_image_pairs(input_folder)
    
    if not pairs:
        print("No matching image pairs found!")
        return
    
    print(f"Found {len(pairs)} image pairs to process...")
    
    # Process each pair
    for index, (base_name, aligned_path, d2_path) in enumerate(pairs, 1):
        # Format index with leading zeros (4 digits)
        index_str = f"{index:04d}"
        
        print(f"Processing pair {index}/{len(pairs)}: {base_name}")
        
        try:
            # Process start image (aligned)
            with Image.open(aligned_path) as img:
                resized_img = resize_image_proportional(img, max_size)
                start_filename = f"{index_str}_start.jpg"
                start_path = output_path / start_filename
                resized_img.save(start_path, 'JPEG', quality=95)
                print(f"  Saved: {start_filename} ({resized_img.size[0]}x{resized_img.size[1]})")
            
            # Process end image (D2)
            with Image.open(d2_path) as img:
                resized_img = resize_image_proportional(img, max_size)
                end_filename = f"{index_str}_end.jpg"
                end_path = output_path / end_filename
                resized_img.save(end_path, 'JPEG', quality=95)
                print(f"  Saved: {end_filename} ({resized_img.size[0]}x{resized_img.size[1]})")
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            continue
    
    print(f"\nProcessing complete! Processed {len(pairs)} image pairs.")
    print(f"Output saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess image pairs for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocess_images.py input_folder output_folder
  python preprocess_images.py input_folder output_folder --max-size 2048
        """
    )
    
    parser.add_argument('input_folder', help='Input folder containing *_D2_aligned.jpg and *_D2.jpg files')
    parser.add_argument('output_folder', help='Output folder for processed images')
    parser.add_argument('--max-size', type=int, default=1536, 
                       help='Maximum size for largest dimension (default: 1536)')
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist!")
        sys.exit(1)
    
    if not os.path.isdir(args.input_folder):
        print(f"Error: '{args.input_folder}' is not a directory!")
        sys.exit(1)
    
    # Process images
    process_images(args.input_folder, args.output_folder, args.max_size)


if __name__ == "__main__":
    main()