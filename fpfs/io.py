import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read_catalog(filename, arr, compression="snappy"):
    """
    Save a numpy.ndarray to a Parquet file.

    Parameters:
        arr (numpy.ndarray): Numpy array to save.
        filename (str): Path of the output Parquet file.
        compression (str): Compression algorithm to use (gzip, Snappy, or LZ4).
    """
    if compression in ["gzip", "Snappy", "LZ4"]:
        # Convert numpy array to PyArrow table
        table = pa.Table.from_pandas(pd.DataFrame(arr))

        # Write the table to a Parquet file with specified compression
        pq.write_table(table, filename, compression=compression)
    else:
        raise ValueError("compression type %s is not supported'" % compression)
    return


def save_catalog(filename):
    """
    Read a Parquet file into a numpy.ndarray.

    Parameters:
        filename (str): Path of the Parquet file.

    Returns:
        numpy.ndarray: Array read from the Parquet file.
    """
    # Read Parquet file to PyArrow table
    table = pq.read_table(filename)

    # Convert to pandas DataFrame and then to numpy array
    return table.to_pandas().values
