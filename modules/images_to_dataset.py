import argparse
from pathlib import Path
import requests
from PIL import Image
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # For progress bars


def download_image(url, download_dir):
    """
    Downloads a single image from a URL and saves it in the specified directory.

    Args:
        url (str): URL of the image to download.
        download_dir (Path): Directory to save the downloaded image.

    Returns:
        Path or None: Path to the downloaded file if successful, otherwise None.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        filename = download_dir / Path(url.split('?')[0]).name
        with open(filename, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return filename
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None


def download_images(csv_file, csv_column, download_dir, num_workers=4):
    """
    Downloads images from URLs provided in a specified CSV column using multithreading.

    Args:
        csv_file (Path): Path to the CSV file containing image URLs.
        csv_column (str): Name of the column in the CSV file that contains URLs.
        download_dir (Path): Directory where the downloaded images will be saved.
        num_workers (int): Number of worker threads for downloading.

    Returns:
        list: A list of file paths for the successfully downloaded images.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    urls = pd.read_csv(csv_file)[csv_column]
    downloaded_files = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(download_image, url, download_dir) for url in urls]
        for future in tqdm(as_completed(futures), total=len(urls), desc="Downloading images"):
            result = future.result()
            if result:
                downloaded_files.append(result)

    return downloaded_files


def gather_input_images(input_dirs):
    """
    Gathers all image files (PNG, JPG) from the provided input directories or files.

    Args:
        input_dirs (list): List of directories or files to gather images from.

    Returns:
        list: A list of file paths to the gathered images.
    """
    input_files = []
    for path in input_dirs:
        p = Path(path)
        if p.is_dir():
            input_files.extend(list(p.glob("**/*.png")) + list(p.glob("**/*.jpg")))
        elif p.is_file() and (p.suffix.lower() in [".png", ".jpg"]):
            input_files.append(p)
    return input_files


def convert_and_filter_image(input_file, output_dir, compression_level, min_side, max_pixels, image_size):
    """
    Processes a single image: crops, converts to PNG, and filters by size criteria.

    Args:
        input_file (Path): Path to the input image.
        output_dir (Path): Directory where the processed image will be saved.
        compression_level (int): PNG compression level (0-9).
        min_side (int): Minimum allowed size of the smaller side.
        max_pixels (int): Maximum total number of pixels.
        image_size (int): Size of the square center crop.

    Returns:
        Path or None: Path to the processed file if successful, otherwise None.
    """
    try:
        with Image.open(input_file) as img:
            width, height = img.size
            smaller_side = min(width, height)
            total_pixels = width * height

            if ((min_side is None or smaller_side >= min_side) and
                    (max_pixels is None or total_pixels <= max_pixels)):
                crop_size = min(image_size, smaller_side)
                left = (img.width - crop_size) // 2
                top = (img.height - crop_size) // 2
                img = img.crop((left, top, left + crop_size, top + crop_size))

                output_file = output_dir / input_file.with_suffix(".png").name
                img.save(output_file, format="PNG", compress_level=compression_level)
                return output_file
    except Exception as e:
        print(f"Failed to process {input_file}: {e}")
    return None


def convert_and_filter_images(input_files, output_dir, compression_level=6, min_side=None, max_pixels=None,
                              image_size=512, num_workers=4):
    """
    Processes images in parallel: crops, converts to PNG, and filters by size criteria.

    Args:
        input_files (list): List of file paths to the downloaded images.
        output_dir (Path): Directory where the processed .png images will be saved.
        compression_level (int): PNG compression level (0-9).
        min_side (int): Minimum allowed size of the smaller side.
        max_pixels (int): Maximum total number of pixels.
        image_size (int): Size of the square center crop.
        num_workers (int): Number of worker threads for processing.

    Returns:
        list: A list of file paths for successfully processed images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_files = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(convert_and_filter_image, file, output_dir, compression_level,
                                   min_side, max_pixels, image_size) for file in input_files]
        for future in tqdm(as_completed(futures), total=len(input_files), desc="Processing images"):
            result = future.result()
            if result:
                processed_files.append(result)

    return processed_files


def main():
    parser = argparse.ArgumentParser(description="Download, convert, crop, and filter images.")
    parser.add_argument("--input_type", type=str, required=True, choices=["csv", "png", "jpg"],
                        help="Type of input data: 'csv' for URLs or 'png'/'jpg' for already-downloaded images.")
    parser.add_argument("--csv_file", type=Path, help="Path to the CSV file containing image URLs.")
    parser.add_argument("--csv_column", type=str, help="Column name in the CSV file containing URLs.")
    parser.add_argument("--input_dirs", nargs="+", type=Path,
                        help="Directories or files containing images to process (for 'png' or 'jpg' input type).")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the processed images.")
    parser.add_argument("--download_dir", type=Path, default=None,
                        help="Directory to save downloaded images. Defaults to 'output_dir/download'.")
    parser.add_argument("--compression", type=int, default=6, help="PNG compression level (0-9).")
    parser.add_argument("--min_side", type=int, default=None, help="Minimum size of the smaller side (optional).")
    parser.add_argument("--max_pixels", type=int, default=None, help="Maximum total number of pixels (optional).")
    parser.add_argument("--image_size", type=int, default=512, help="Size of the square center crop (default: 512 px).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for parallel execution.")

    args = parser.parse_args()

    # Handle input types
    if args.input_type == "csv":
        if not args.csv_file or not args.csv_column:
            raise argparse.ArgumentTypeError("For input_type 'csv', both --csv_file and --csv_column are required.")
        download_dir = args.download_dir if args.download_dir else args.output_dir / "download"
        print("Downloading images...")
        input_files = download_images(args.csv_file, args.csv_column, download_dir, args.num_workers)
    else:  # png or jpg
        if not args.input_dirs:
            raise argparse.ArgumentTypeError("For input_type 'png' or 'jpg', --input_dirs is required.")
        input_files = gather_input_images(args.input_dirs)

    print("Processing images...")
    convert_and_filter_images(input_files, args.output_dir, args.compression, args.min_side,
                              args.max_pixels, args.image_size, args.num_workers)


if __name__ == "__main__":
    main()
