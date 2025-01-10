import pandas as pd
import argparse

def concatenate_parquet_files(file1, file2, output_file):
    """
    Concatenate two Parquet files into one.

    Parameters:
        file1 (str): Path to the first Parquet file.
        file2 (str): Path to the second Parquet file.
        output_file (str): Path to the output Parquet file.

    Returns:
        None
    """
    try:
        # Read the Parquet files into Pandas DataFrames
        df1 = pd.read_parquet(file1)
        df2 = pd.read_parquet(file2)

        # Concatenate the DataFrames
        concatenated_df = pd.concat([df1, df2], ignore_index=True)

        # Write the concatenated DataFrame to a new Parquet file
        concatenated_df.to_parquet(output_file, index=False)

        print(f"Successfully concatenated files into: {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Main function to parse arguments and run the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate two Parquet files into one.")
    parser.add_argument("--file1", type=str, required=True, help="Path to the first Parquet file.")
    parser.add_argument("--file2", type=str, required=True, help="Path to the second Parquet file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output Parquet file.")

    args = parser.parse_args()

    concatenate_parquet_files(args.file1, args.file2, args.output_file)
