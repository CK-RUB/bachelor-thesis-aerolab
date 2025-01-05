import argparse
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from urllib.parse import urlparse


def download_image(url, download_dir, suppress_prints=False):
    """
    Downloads a single image from a URL and saves it in the specified directory.

    Args:
        url (str): URL of the image to download.
        download_dir (Path): Directory to save the downloaded image.
        suppress_prints (bool): If True, suppresses all prints.

    Returns:
        Path or None: Path to the downloaded file if successful, otherwise None.
    """

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Extract hostname from URL
        hostname = urlparse(url).hostname or "unknown"

        # Create a subdirectory for the hostname
        hostname_dir = download_dir / hostname
        hostname_dir.mkdir(parents=True, exist_ok=True)

        # Extract filename without query parameters
        filename = hostname_dir / Path(url.split('?')[0]).name

        # Filter by extension before saving
        if filename.suffix.lower() not in Image.registered_extensions():
            if not suppress_prints:
                print(f"Unsupported file type for URL {url}: {filename.suffix}")

            return None

        # Check if the file already exists and append a number if necessary
        counter = 1

        while filename.exists():
            filename = hostname_dir / f"{filename.stem}_{counter}{filename.suffix}"
            counter += 1

        with open(filename, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        return filename
    except Exception as e:
        if not suppress_prints:
            print(f"Failed to download {url}: {e}")

        return None


def download_images(file_type, input_file, column_name, download_dir, num_workers, n_first, suppress_prints=False):
    """
    Downloads images from URLs provided in a TXT or CSV file using multithreading.

    Args:
        file_type (str): Type of the input file ('txt' or 'csv').
        input_file (Path): Path to the TXT or CSV file containing image URLs.
        column_name (str): Name of the column in the CSV file that contains URLs (only used if file_type is 'csv').
        download_dir (Path): Directory where the downloaded images will be saved.
        num_workers (int): Number of worker threads for downloading.
        n_first (int): If specified, only download the first n images.
        suppress_prints (bool): If True, suppresses all prints.

    Returns:
        list: A list of file paths for the successfully downloaded images.
    """

    download_dir.mkdir(parents=True, exist_ok=True)

    if file_type == "txt":
        urls = input_file.read_text().splitlines()[:n_first] if n_first else input_file.read_text().splitlines()
    else:
        urls = pd.read_csv(input_file)[column_name][:n_first] if n_first else pd.read_csv(input_file)[column_name]

    downloaded_files = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(download_image, url, download_dir, suppress_prints) for url in urls]

        for future in tqdm(as_completed(futures), total=len(urls), desc="Downloading images", disable=suppress_prints):
            result = future.result()

            if result:
                downloaded_files.append(result)

    return downloaded_files


def gather_local_images(input_dirs, n_first, suppress_prints=False):
    """
    Gathers all supported image files from the provided input directories or files.

    Args:
        input_dirs (list): List of directories or files to gather images from.
        n_first (int): If specified, only gather the first n images.
        suppress_prints (bool): If True, suppresses all prints.

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
            if not suppress_prints:
                print(f"Unsupported file type for file {f}: {f.suffix}")

    return valid_files[:n_first] if n_first else valid_files


def process_image(input_file, output_file, compression, min_side, max_pixels, image_size, suppress_prints=False):
    """
    Processes a single image: crops, converts, and filters by size criteria.

    Args:
        input_file (Path): Path to the input image.
        output_file (Path): Path to save the processed image.
        compression (int): Compression level (0-95 for JPG, 0-9 for PNG).
        min_side (int): Minimum allowed size of the smaller side.
        max_pixels (int): Maximum total number of pixels.
        image_size (int): Size of the square center crop.
        suppress_prints (bool): If True, suppresses all prints.

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
        if not suppress_prints:
            print(f"Failed to process {input_file}: {e}")

    return None


def process_images(input_files, input_dirs, output_type, output_dir, compression, min_side, max_pixels,
                   image_size, num_workers, suppress_prints=False):
    """
    Processes images in parallel: crops, converts, and filters by size criteria.

    Args:
        input_files (list): List of file paths to the input images.
        input_dirs (list): List of input directories for calculating relative paths.
        output_type (str): Output image format ('png' or 'jpg').
        output_dir (Path): Directory where the processed images will be saved.
        compression (int): Compression level (0-95 for JPG, 0-9 for PNG).
        min_side (int): Minimum allowed size of the smaller side.
        max_pixels (int): Maximum total number of pixels.
        image_size (int): Size of the square center crop.
        num_workers (int): Number of worker threads for processing.
        suppress_prints (bool): If True, suppresses all prints.

    Returns:
        list: A list of Path objects for the directories containing the successfully processed images.
    """

    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_dirs = []


    def compute_relative_path(filepath):
        """
        Computes the relative path of a file with respect to the provided input directories.

        Args:
            filepath (Path): The file path to compute the relative path for.

        Returns:
            Path: The relative path if the file is within any input directory, otherwise the file name.
        """

        for input_dir in input_dirs:
            input_dir_path = Path(input_dir)

            if filepath.is_relative_to(input_dir_path):
                return input_dir_path.name / filepath.relative_to(input_dir_path)

        return filepath.name


    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures_to_dirs = {}

        for file in input_files:
            relative_path = compute_relative_path(file)
            specific_output_dir = dataset_dir / relative_path.parent
            specific_output_dir.mkdir(parents=True, exist_ok=True)

            futures = executor.submit(
                process_image,
                file,
                specific_output_dir / file.with_suffix(f".{output_type}").name,
                compression,
                min_side,
                max_pixels,
                image_size,
                suppress_prints
            )

            futures_to_dirs[futures] = specific_output_dir

        for future in tqdm(as_completed(futures_to_dirs), total=len(futures_to_dirs), desc="Processing images", disable=suppress_prints):
            result = future.result()
            specific_output_dir = futures_to_dirs[future]

            if result and specific_output_dir not in dataset_dirs:
                dataset_dirs.append(specific_output_dir)

    return dataset_dirs


def raise_value_error(error_message, cli_context=True):
    if cli_context:
        raise argparse.ArgumentTypeError(error_message)
    else:
        raise ValueError(error_message)


def create_dataset(input_types, input_csvs, csv_columns, input_txts, input_dirs, output_type, output_dir, download_dir,
                   compression, min_side, max_pixels, image_size=512, num_workers=64, download_only=False, n_first=None,
                   suppress_prints=False, cli_context=False):
    images_for_processing = []
    input_dirs_for_processing = []

    if compression is None:
        compression = 75 if output_type == "jpg" else 6
    else:
        if output_type == "jpg":
            if not 0 <= compression <= 95:
                raise_value_error("Compression level must be between 0 and 95 for JPG.", cli_context=cli_context)
        if output_type == "png":
            if not 0 <= compression <= 9:
                raise_value_error("Compression level must be between 0 and 9 for PNG.", cli_context=cli_context)

    download_dir = download_dir if download_dir else output_dir / "download"

    if "txt" in input_types:
        if not input_txts:
            raise_value_error(f"For input_type 'txt', --input_txts is required.", cli_context=cli_context)

        if not suppress_prints:
            print(f"Downloading images from {input_types.upper()}...")

        for input_txt in input_txts:
            images_for_processing.extend(download_images("txt", input_txt, "", download_dir, num_workers, n_first, suppress_prints))

    if "csv" in input_types:
        if not input_csvs:
            raise_value_error(f"For input_type 'csv', --input_csvs is required.", cli_context=cli_context)

        if not csv_columns:
            raise_value_error(f"For input_type 'csv', --csv_columns is required.", cli_context=cli_context)

        if len(input_csvs) != len(csv_columns):
            raise_value_error(f"Number of --input_csvs and --csv_columns must be the same.", cli_context=cli_context)

        if not suppress_prints:
            print(f"Downloading images from {input_types.upper()}...")

        for input_csv, column_name in zip(input_csvs, csv_columns):
            images_for_processing.extend(download_images("csv", input_csv, column_name, download_dir, num_workers, n_first, suppress_prints))

    if input_types in ["txt", "csv"]:
        input_dirs_for_processing.extend([download_dir])

        if not suppress_prints:
            print("Downloaded images saved to:", download_dir)

    if download_only:
        if not suppress_prints:
            print("Download only mode enabled. Skipping processing.")

        return

    if "file" in input_types:
        if not input_dirs:
            raise_value_error(f"For input_type 'file', --input_dirs is required.", cli_context=cli_context)

        if not suppress_prints:
            print("Gathering input images...")

        images_for_processing.extend(gather_local_images(input_dirs, n_first, suppress_prints))
        input_dirs_for_processing.extend(input_dirs)

    if not suppress_prints:
        print("Processing images...")

    dataset_dirs = process_images(
        input_files=images_for_processing,
        input_dirs=input_dirs_for_processing,
        output_type=output_type,
        output_dir=output_dir,
        compression=compression,
        min_side=min_side,
        max_pixels=max_pixels,
        image_size=image_size,
        num_workers=num_workers,
        suppress_prints=suppress_prints
    )

    if not suppress_prints:
        print("Processed Dataset Directories:")
        print("-----------------------------")

    for directory in dataset_dirs:
        print(directory)

    return dataset_dirs


def main():
    parser = argparse.ArgumentParser(description="Download, convert, crop, and filter images.")

    # Input Args
    parser.add_argument("--input_types", nargs="+", type=str, required=True, choices=["csv", "txt", "file"],
                        help="Type of input data. Choices: 'csv', 'txt', 'file'. Required.")
    parser.add_argument("--input_csvs", nargs="+", type=Path, required=False,
                        help="Path to CSV files containing image URLs. Required for 'csv'.")
    parser.add_argument("--csv_columns", nargs="+", type=str, required=False,
                        help="Column names in the CSV files containing URLs. Required for 'csv'.")
    parser.add_argument("--input_txts", nargs="+", type=Path, required=False,
                        help="Path to the TXT or CSV file containing image URLs. Required for 'txt'.")
    parser.add_argument("--input_dirs", nargs="+", type=Path, required=False,
                        help="Directories or files containing images to process. Required for 'file'.")

    # Output Args
    parser.add_argument("--output_type", type=str, default="png", required=False, choices=["png", "jpg"],
                        help="Output image format. Choices: 'png', 'jpg'. Default: 'png'. Optional.")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory to save the processed images. Required.")
    parser.add_argument("--download_dir", type=Path, default=None, required=False,
                        help="Directory to save downloaded images. Defaults to 'output_dir/download'. Optional.")

    # Processing Args
    parser.add_argument("--compression", type=int, default=None, required=False,
                        help="Compression level. Range: 0-95 for JPG or 0-9 for PNG. Default: 75 or 6. Optional.")
    parser.add_argument("--min_side", type=int, default=None, required=False,
                        help="Minimum size of the smaller side. Default: None. Optional.")
    parser.add_argument("--max_pixels", type=int, default=None, required=False,
                        help="Maximum total number of pixels. Default: None. Optional.")
    parser.add_argument("--image_size", type=int, default=512, required=False,
                        help="Size of the square center crop. Default: 512 pixels. Optional.")

    # Execution Args
    parser.add_argument("--num_workers", type=int, default=64, required=False,
                        help="Number of worker threads for parallel execution. Default: 64. Optional.")
    parser.add_argument("--download_only", action="store_true", required=False, default=False,
                        help="If set, only downloads images and skips processing. Applicable for 'csv' or 'txt'. Optional.")
    parser.add_argument("--n_first", type=int, default=None, required=False,
                        help="If specified, restricts downloading and processing to the first n images. Optional.")
    parser.add_argument("--supress_prints", action="store_true", required=False, default=False,
                        help="If set, suppresses all prints. Optional.")

    args = parser.parse_args()

    create_dataset(
        input_types=args.input_types,
        input_csvs=args.input_csvs,
        csv_columns=args.csv_columns,
        input_txts=args.input_txts,
        input_dirs=args.input_dirs,
        output_type=args.output_type,
        output_dir=args.output_dir,
        download_dir=args.download_dir,
        compression=args.compression,
        min_side=args.min_side,
        max_pixels=args.max_pixels,
        image_size=args.image_size,
        num_workers=args.num_workers,
        download_only=args.download_only,
        n_first=args.n_first,
        suppress_prints=args.supress_prints,
        cli_context=True
    )


if __name__ == "__main__":
    main()
