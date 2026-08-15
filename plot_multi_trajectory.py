import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate P and Q error plots from a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the input CSV file")
    parser.add_argument("--ignore", "-i", type=int, nargs="+", default=[], 
                        help="List of trajectory numbers to ignore in the plots (e.g., -i 1 2)")
    parser.add_argument("--logscale", action="store_true", 
                        help="Plot the errors on a logarithmic scale")
    parser.add_argument("--separate", "-s", action="store_true", 
                        help="Create separate plots for 1-step and 2-step methods")
    parser.add_argument("--outdir", "-o", type=str, default=".", 
                        help="Directory to save the plots. Will be created if it does not exist.")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Extract the file name without the extension
    file_stem = Path(args.csv_file).stem

    # Handle output directory creation
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{args.csv_file}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)

    # Remove the ignored rows
    if args.ignore:
        indices_to_ignore = [t - 1 for t in args.ignore]
        df = df.drop(index=indices_to_ignore, errors='ignore')
        print(f"Ignored trajectories: {args.ignore}")

    # Define x-axis values (from 1 to 10 trajectories, adjusting for ignored indices)
    x = df.index + 1
    
    # Determine the filename suffix based on the logscale argument
    suffix = "_logscale" if args.logscale else ""

    if args.separate:
        # ==========================================
        # Separate Plots (4 total)
        # ==========================================
        plots_info = [
            ('1S P RE Mean', '1S P RE STD', '1-Step Error (P)', 'blue', 'o', f"{file_stem}_1s_p_error_plot{suffix}.png", 'P'),
            ('2S P RE Mean', '2S P RE STD', '2-Step Error (P)', 'red', 's', f"{file_stem}_2s_p_error_plot{suffix}.png", 'P'),
            ('1S Q RE Mean', '1S Q RE STD', '1-Step Error (Q)', 'green', 'o', f"{file_stem}_1s_q_error_plot{suffix}.png", 'Q'),
            ('2S Q RE Mean', '2S Q RE STD', '2-Step Error (Q)', 'orange', 's', f"{file_stem}_2s_q_error_plot{suffix}.png", 'Q')
        ]

        for mean_col, std_col, label, color, marker, filename, var_name in plots_info:
            plt.figure(figsize=(10, 6))
            plt.plot(x, df[mean_col], label=label, color=color, marker=marker)
            plt.fill_between(x, df[mean_col] - df[std_col], 
                             df[mean_col] + df[std_col], color=color, alpha=0.2)
            
            plt.xlabel('Number of Trajectories used for training')
            plt.ylabel(f'Relative Error ({var_name})')
            plt.title(f'{label}')
            plt.xticks(x)
            
            if args.logscale:
                plt.yscale('log')
                
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            # Save plot to outdir
            out_path = outdir / filename
            plt.savefig(out_path)
            plt.close()
            print(f"Successfully generated '{out_path}'")

    else:
        # ==========================================
        # Combined Plots (2 total)
        # ==========================================
        
        # --- Plot 1: P Error Comparison ---
        plt.figure(figsize=(10, 6))

        plt.plot(x, df['1S P RE Mean'], label='1-Step Error (P)', color='blue', marker='o')
        plt.fill_between(x, df['1S P RE Mean'] - df['1S P RE STD'], 
                         df['1S P RE Mean'] + df['1S P RE STD'], color='blue', alpha=0.2)

        plt.plot(x, df['2S P RE Mean'], label='2-Step Error (P)', color='red', marker='s')
        plt.fill_between(x, df['2S P RE Mean'] - df['2S P RE STD'], 
                         df['2S P RE Mean'] + df['2S P RE STD'], color='red', alpha=0.2)

        plt.xlabel('Number of Trajectories used for training')
        plt.ylabel('Relative Error (P)')
        plt.title('Comparison of 1-Step and 2-Step Errors on P')
        plt.xticks(x)
        
        if args.logscale:
            plt.yscale('log')
            
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # Save plot to outdir
        p_plot_name = f"{file_stem}_p_error_plot{suffix}.png"
        out_path_p = outdir / p_plot_name
        plt.savefig(out_path_p)
        plt.close()
        print(f"Successfully generated '{out_path_p}'")

        # --- Plot 2: Q Error Comparison ---
        plt.figure(figsize=(10, 6))

        plt.plot(x, df['1S Q RE Mean'], label='1-Step Error (Q)', color='green', marker='o')
        plt.fill_between(x, df['1S Q RE Mean'] - df['1S Q RE STD'], 
                         df['1S Q RE Mean'] + df['1S Q RE STD'], color='green', alpha=0.2)

        plt.plot(x, df['2S Q RE Mean'], label='2-Step Error (Q)', color='orange', marker='s')
        plt.fill_between(x, df['2S Q RE Mean'] - df['2S Q RE STD'], 
                         df['2S Q RE Mean'] + df['2S Q RE STD'], color='orange', alpha=0.2)

        plt.xlabel('Number of Trajectories used for training')
        plt.ylabel('Relative Error (Q)')
        plt.title('Comparison of 1-Step and 2-Step Errors on Q')
        plt.xticks(x)
        
        if args.logscale:
            plt.yscale('log')
            
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # Save plot to outdir
        q_plot_name = f"{file_stem}_q_error_plot{suffix}.png"
        out_path_q = outdir / q_plot_name
        plt.savefig(out_path_q)
        plt.close()
        print(f"Successfully generated '{out_path_q}'")

if __name__ == "__main__":
    main()