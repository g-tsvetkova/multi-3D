# plot_inference_times.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def plot_inference_times(csv_file, output_dir='.', output_prefix='inference_time'):
    """
    Reads the inference times from a CSV file, converts them to minutes,
    and plots a line chart comparing different models over iterations.

    Parameters:
    - csv_file (str): Path to the CSV file containing inference times.
    - output_dir (str): Directory where the plot images will be saved.
    - output_prefix (str): Prefix for the saved plot image filenames.
    """
    # Check if the CSV file exists
    if not os.path.isfile(csv_file):
        print(f"Error: The file '{csv_file}' does not exist in the specified path.")
        return

    # Ensure the output directory exists; if not, create it
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Validate the required columns
    required_columns = {'Model', 'Iteration', 'Inference_Time_sec'}
    if not required_columns.issubset(df.columns):
        print(f"Error: The CSV file must contain the following columns: {required_columns}")
        return

    # Convert Iteration to integer if it's not
    df['Iteration'] = df['Iteration'].astype(int)

    # Convert Inference_Time_sec to minutes for better readability
    df['Inference_Time_min'] = df['Inference_Time_sec'] / 60

    # Set the Seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Initialize the matplotlib figure
    plt.figure(figsize=(12, 8))

    # Create a line plot with markers for each model
    sns.lineplot(
        data=df,
        x='Iteration',
        y='Inference_Time_min',
        hue='Model',
        marker='o',
        palette='viridis'
    )

    # Add title and labels
    plt.title('Inference Time Comparison Between Models', fontsize=16)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Inference Time (minutes)', fontsize=14)

    # Customize legend
    plt.legend(title='Model', fontsize=12, title_fontsize=14)

    # Annotate each point with its exact inference time in minutes
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        for _, row in model_data.iterrows():
            plt.text(
                row['Iteration'],
                row['Inference_Time_min'] + 0.02,  # Slightly above the point
                f"{row['Inference_Time_min']:.2f}",
                horizontalalignment='center',
                fontsize=9
            )

    # Adjust layout for better spacing
    plt.tight_layout()

    # Define the output path for the plot
    plot_path = os.path.join(output_dir, f"{output_prefix}_line_chart.png")

    # Save the plot to a file
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved as '{plot_path}'.")

    # Display the plot
    plt.show()

    # Optional: Print a summary of average times in minutes
    summary = df.groupby('Model')['Inference_Time_min'].agg(['mean', 'std']).reset_index()
    print("\nAverage Inference Times (minutes):")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Plot Inference Times from CSV')
    parser.add_argument(
        '--csv',
        type=str,
        default='inference_times.csv',
        help='Path to the CSV file containing inference times (default: inference_times.csv)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save the output plots (default: current directory)'
    )
    parser.add_argument(
        '--output_prefix',
        type=str,
        default='inference_time',
        help='Prefix for the saved plot image filenames (default: inference_time)'
    )
    args = parser.parse_args()

    # Call the plotting function with parsed arguments
    plot_inference_times(args.csv, args.output_dir, args.output_prefix)
