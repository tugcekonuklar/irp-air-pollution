import os
import pyarrow.parquet as pq


def convert_parquet_to_csv(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".parquet"):
            parquet_file = os.path.join(input_folder, filename)
            csv_file = os.path.join(output_folder, filename.replace(".parquet", ".csv"))
            try:
                # Read the Parquet file into a DataFrame
                parquet_table = pq.read_table(parquet_file)
                df = parquet_table.to_pandas()

                # Write the DataFrame to a CSV file
                df.to_csv(csv_file, index=False)

                print(f"Conversion from {parquet_file} to {csv_file} successful.")
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    input_folder = "data/parquet"  # Replace with the path to your data folder
    output_folder = "data/csv"  # Replace with the path to your output CSV folder

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Call the conversion function
    convert_parquet_to_csv(input_folder, output_folder)
