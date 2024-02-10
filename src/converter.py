import os
import pyarrow.parquet as pq
import pandas as pd


def convert_parquet_to_csv(input_folder, output_folder):
    """
    Converts all Parquet files in the input_folder to CSV files in the output_folder.

    Parameters:
    - input_folder: Folder containing Parquet files.
    - output_folder: Destination folder for CSV files.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".parquet"):
            # Full paths for the input Parquet and output CSV files
            parquet_file = os.path.join(input_folder, filename)
            csv_file = os.path.join(output_folder, filename.replace(".parquet", ".csv"))

            try:
                # Read the Parquet file and convert to a DataFrame
                df = pq.read_table(parquet_file).to_pandas()

                # Write the DataFrame to a CSV file
                df.to_csv(csv_file, index=False)
                print(f"Successfully converted {parquet_file} to {csv_file}.")
            except Exception as e:
                print(f"Error converting {parquet_file}: {e}")


# Example usage
if __name__ == "__main__":
    input_folder = "data/parquet"  # Path to your Parquet files
    output_folder = "data/csv"  # Path for the resulting CSV files
    convert_parquet_to_csv(input_folder, output_folder)
