import argparse
from pathlib import Path
import requests
from PIL import Image
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os


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

        # Extract filename without query parameters
        filename = download_dir / Path(url.split('?')[0]).name

        # Filter by extension before saving
        if filename.suffix.lower() not in Image.registered_extensions():
            print(f"Unsupported file type for URL {url}: {filename.suffix}")

            return None

        # Check if the file already exists and append a number if necessary
        counter = 1

        while filename.exists():
            filename = download_dir / f"{filename.stem}_{counter}{filename.suffix}"

        with open(filename, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        return filename
    except Exception as e:
        print(f"Failed to download {url}: {e}")

        return None


def download_images(input_file, file_type, column_name, download_dir, num_workers, n_first):
    """
    Downloads images from URLs provided in a TXT or CSV file using multithreading.

    Args:
        input_file (Path): Path to the TXT or CSV file containing image URLs.
        file_type (str): Type of the input file ('txt' or 'csv').
        column_name (str): Name of the column in the CSV file that contains URLs (only used if file_type is 'csv').
        download_dir (Path): Directory where the downloaded images will be saved.
        num_workers (int): Number of worker threads for downloading.
        n_first (int): If specified, only download the first n images.

    Returns:
        list: A list of file paths for the successfully downloaded images.
    """

    download_dir.mkdir(parents=True, exist_ok=True)

    if file_type == "txt":
        urls = input_file.read_text().splitlines()[:n_first] if n_first else input_file.read_text().splitlines()
    elif file_type == "csv":
        urls = pd.read_csv(input_file)[column_name][:n_first] if n_first else pd.read_csv(input_file)[column_name]
    else:
        raise ValueError("Unsupported file type. Only 'txt' and 'csv' are allowed.")

    downloaded_files = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(download_image, url, download_dir) for url in urls]

        for future in tqdm(as_completed(futures), total=len(urls), desc="Downloading images"):
            result = future.result()

            if result:
                downloaded_files.append(result)

    return downloaded_files


def gather_local_images(input_dirs, n_first):
    """
    Gathers all supported image files from the provided input directories or files.

    Args:
        input_dirs (list): List of directories or files to gather images from.
        n_first (int): If specified, only gather the first n images.

    Returns:
        list: A list of file paths to the gathered images.
    """

    input_files = []

    for path in input_dirs:
        p = Path(path)

        if p.is_dir():
            input_files.extend(list(p.rglob("*")))
        elif p.is_file():
            input_files.append(p)

    valid_files = []

    for f in input_files:
        if f.suffix.lower() in Image.registered_extensions():
            valid_files.append(f)
        else:
            print(f"Unsupported file type for file {f}: {f.suffix}")

    return valid_files[:n_first] if n_first else valid_files

def process_image(input_file, output_file, compression, min_side, max_pixels, image_size):
    """
    Processes a single image: crops, converts, and filters by size criteria.

    Args:
        input_file (Path): Path to the input image.
        output_file (Path): Path to save the processed image.
        compression (int): Compression level (0-100 for JPG, 0-9 for PNG).
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

                img.save(output_file, quality=compression if output_file.suffix == ".jpg" else None,
                         compress_level=compression if output_file.suffix == ".png" else None)

                return output_file
    except Exception as e:
        print(f"Failed to process {input_file}: {e}")

    return None


def process_images(input_files, output_dir, output_type, compression, min_side, max_pixels,
                   image_size, num_workers):
    """
    Processes images in parallel: crops, converts, and filters by size criteria.

    Args:
        input_files (list): List of file paths to the input images.
        output_dir (Path): Directory where the processed images will be saved.
        output_type (str): Output image format ('png' or 'jpg').
        compression (int): Compression level (0-100 for JPG, 0-9 for PNG).
        min_side (int): Minimum allowed size of the smaller side.
        max_pixels (int): Maximum total number of pixels.
        image_size (int): Size of the square center crop.
        num_workers (int): Number of worker threads for processing.

    Returns:
        list: A list of file paths for successfully processed images.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_files = []

    # Determine the common base directory
    common_base = os.path.commonpath([str(file.parent) for file in input_files])

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for file in input_files:
            relative_subdir = Path(file.parent).relative_to(common_base)
            specific_output_dir = output_dir / relative_subdir

            if specific_output_dir.exists():
                specific_output_dir = specific_output_dir.with_name(f"{specific_output_dir.name}-dataset")

            specific_output_dir.mkdir(parents=True, exist_ok=True)
            futures.append(
                executor.submit(
                    process_image,
                    file,
                    specific_output_dir / file.with_suffix(f".{output_type}").name,
                    compression,
                    min_side,
                    max_pixels,
                    image_size
                )
            )

        for future in tqdm(as_completed(futures), total=len(input_files), desc="Processing images"):
            result = future.result()

            if result:
                processed_files.append(result)

    return processed_files


def main():
    parser = argparse.ArgumentParser(description="Download, convert, crop, and filter images.")

    parser.add_argument("--input_type", type=str, required=True, choices=["csv", "txt", "file"],
                        help="Type of input data. Choices: 'csv', 'txt', 'file'. Required.")
    parser.add_argument("--input_file", type=Path, required=False,
                        help="Path to the TXT or CSV file containing image URLs. Required for 'csv' or 'txt'.")
    parser.add_argument("--csv_column", type=str, required=False,
                        help="Column name in the CSV file containing URLs. Required for 'csv'.")
    parser.add_argument("--input_dirs", nargs="+", type=Path, required=False,
                        help="Directories or files containing images to process. Required for 'file'.")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory to save the processed images. Required.")
    parser.add_argument("--download_dir", type=Path, required=False, default=None,
                        help="Directory to save downloaded images. Defaults to 'output_dir/download'. Optional.")
    parser.add_argument("--output_type", type=str, required=True, choices=["png", "jpg"],
                        help="Output image format. Choices: 'png', 'jpg'. Required.")
    parser.add_argument("--compression", type=int, required=True,
                        help="Compression level. Range: 0-100 for JPG or 0-9 for PNG. Required.")
    parser.add_argument("--min_side", type=int, default=None, required=False,
                        help="Minimum size of the smaller side. Default: None. Optional.")
    parser.add_argument("--max_pixels", type=int, default=None, required=False,
                        help="Maximum total number of pixels. Default: None. Optional.")
    parser.add_argument("--image_size", type=int, default=512, required=False,
                        help="Size of the square center crop. Default: 512 pixels. Optional.")
    parser.add_argument("--num_workers", type=int, default=64, required=False,
                        help="Number of worker threads for parallel execution. Default: 64. Optional.")
    parser.add_argument("--download_only", action="store_true", required=False, default=False,
                        help="If set, only downloads images and skips processing. Applicable for 'csv' or 'txt'. Optional.")
    parser.add_argument("--n_first", type=int, required=False, default=None,
                        help="If specified, restricts downloading and processing to the first n images. Optional.")

    args = parser.parse_args()

    if args.input_type in ["csv", "txt"]:
        if not args.input_file:
            raise argparse.ArgumentTypeError(f"For input_type '{args.input_type}', --input_file is required.")

        if args.input_type == "csv" and not args.csv_column:
            raise argparse.ArgumentTypeError("For input_type 'csv', --csv_column is required.")

        download_dir = args.download_dir if args.download_dir else args.output_dir / "download"
        print(f"Downloading images from {args.input_type.upper()}...")
        input_files = download_images(args.input_file, args.input_type, args.csv_column, download_dir, args.num_workers, args.n_first)

        if args.download_only:
            print("Download only mode enabled. Skipping processing.")

            return
    else:  # input_type == "file"
        if not args.input_dirs:
            raise argparse.ArgumentTypeError("For input_type 'file', --input_dirs is required.")

        print("Gathering input images...")
        input_files = gather_local_images(args.input_dirs, args.n_first)

    print("Processing images...")
    process_images(input_files, args.output_dir, args.output_type, args.compression, args.min_side,
                   args.max_pixels, args.image_size, args.num_workers)


if __name__ == "__main__":
    main()
