import os
import sys
import argparse
from pillow import Image
from concurrent.futures import ThreadPoolExecutor

# Function to convert and compress an image
def convert_and_compress(input_path, output_path, quality=85):
    try:
        # Open the image
        img = Image.open(input_path)
        
        # Convert and save the image
        img = img.convert("RGB")
        img.save(output_path, optimize=True, quality=quality)
        
        print(f"Converted and compressed: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert and compress image files.")
    parser.add_argument("input_dir", help="Input directory containing image files.")
    parser.add_argument("output_dir", help="Output directory for converted and compressed images.")
    parser.add_argument("--quality", type=int, default=85, help="JPEG compression quality (0-100, higher is better).")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for concurrent processing.")
    
    args = parser.parse_args()

    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # List all image files in the input directory
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ome.tif'))]

    # Create a thread pool for concurrent processing
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        for image_file in image_files:
            input_path = os.path.join(args.input_dir, image_file)
            output_path = os.path.join(args.output_dir, image_file)
            
            # Submit a task to the thread pool
            executor.submit(convert_and_compress, input_path, output_path, args.quality)

if __name__ == "__main__":
    main()