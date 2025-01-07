from typing import Literal, Dict
import matplotlib.pyplot as plt
import numpy as np


def normalize_column_of_nparrays(df, column):
    """
    Normalize all numpy arrays in a specified DataFrame column to the range [0, 1].

    Parameters:
        df (pd.DataFrame): The DataFrame containing the column to normalize.
        column (str): The name of the column with numpy arrays to normalize.

    Returns:
        pd.DataFrame: A copy of the DataFrame with normalized numpy arrays in the specified column.

    Raises:
        ValueError: If all values in the arrays are constant, making normalization impossible.
    """

    all_values = np.stack(df[column]).flatten()

    min_val = all_values.min()
    max_val = all_values.max()

    def normalize_nparray(array):
        if min_val == max_val:
            raise ValueError("Cannot normalize array with constant values.")

        return (array - min_val) / (max_val - min_val)

    df = df.copy()
    df[column] = df[column].apply(normalize_nparray)

    return df

def get_nice_name(input_string: str, custom_mappings: Dict[str, str]) -> str:
    """
    Map an input string to a custom name based on key-value mappings.

    Parameters:
        input_string (str): The string to map.
        custom_mappings (Dict[str, str]): A dictionary where keys are substrings to search for,
                                          and values are the corresponding mapped names.

    Returns:
        str: The mapped name if a key is found in the input string; otherwise, None.
    """

    for key, value in custom_mappings.items():
        if key in input_string:
            return value

def configure_default_plot_style() -> None:
    """
    Configure Matplotlib global settings for consistent and publication-ready plots.

    This function sets up the global Matplotlib parameters to ensure figures adhere 
    to publication standards, including LaTeX text rendering, appropriate font sizes, 
    and figure layouts.
    """
    # Reset to default settings
    plt.rcdefaults()

    # General parameters for styling
    params = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amssymb} \usepackage{amsmath}",
        "font.family": "serif",
        "axes.labelsize": 8,
        "font.size": 8,
        "legend.fontsize": 6,
        "legend.handlelength": 1.0,
        "legend.columnspacing": 1.0,
        "legend.handletextpad": 0.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.labelpad": 2.0,
        "xtick.major.pad": 1.0,
        "ytick.major.pad": 1.0,
        "lines.linewidth": 0.75,
        "lines.markersize": 2,
    }
    plt.rcParams.update(params)

    # Figure-specific parameters
    figure_params = {
        "figure.dpi": 300,
        "figure.constrained_layout.use": True,
        "axes.grid": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.0,
    }
    plt.rcParams.update(figure_params)

    # Set default figure size
    set_default_figsize()

def set_default_figsize(
    format: Literal["single", "double"] = "single",
    ratio: float = 2 / (1 + 5**0.5),  # Default to golden ratio
    factor: float = 1.0,
    nrows: int = 1,
    ncols: int = 1,
) -> None:
    """
    Configure the default width and height of Matplotlib figures.

    This function calculates and sets the figure size based on the given format, 
    aspect ratio, scaling factor, and subplot configuration.

    Args:
        format (Literal["single", "double"]):
            Determines the figure width. "single" for single-column figures, 
            "double" for double-column figures. Defaults to "single".
        ratio (float):
            Aspect ratio of the figure (height = width * ratio). 
            Defaults to the golden ratio.
        factor (float):
            Scaling factor for both width and height. Defaults to 1.0.
        nrows (int):
            Number of rows of subplots. Defaults to 1.
        ncols (int):
            Number of columns of subplots. Defaults to 1.

    Raises:
        ValueError: If the provided format is not "single" or "double".
    """
    if format == "single":
        width = 237.13594  # Single-column width in points
    elif format == "double":
        width = 496.85625  # Double-column width in points
    else:
        raise ValueError("Invalid format. Use 'single' or 'double'.")

    # Calculate figure height based on aspect ratio and subplot configuration
    height = width * ratio * (nrows / ncols)

    # Convert from points to inches and apply scaling factor
    factor = 100 / 7227 * factor
    plt.rcParams["figure.figsize"] = (width * factor, height * factor)
