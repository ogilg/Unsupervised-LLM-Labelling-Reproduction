"""
Plot comparison results matching Figure 1 from the paper.
"""
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Colors from Figure 1 (matching paper)
COLORS = {
    'zero_shot': '#9B6B9E',  # Purple
    'zero_shot_chat': '#9B6B9E',  # Purple with dots pattern
    'golden': '#E8A25A',  # Orange
    'human': '#7FB685',  # Green
    'unsupervised': '#6DAEDB',  # Blue/teal
    'icm_fixed': '#5A9BD4'  # Slightly different blue for fixed context
}


def load_results(results_file):
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_comparison(results_file, output_file=None, include_fixed=False,
                   fixed_results_file=None):
    """
    Create comparison plot matching Figure 1 from paper.

    Args:
        results_file: Path to comparison results JSON
        output_file: Where to save the plot (auto-generated if None)
        include_fixed: Whether to include fixed-context ICM as 5th bar
        fixed_results_file: Path to fixed-context results JSON (if include_fixed=True)
    """
    results = load_results(results_file)

    # Extract accuracies (convert to percentages)
    zero_shot_base = results['results']['zero_shot_base'] * 100
    zero_shot_instruct = results['results']['zero_shot_instruct'] * 100
    golden = results['results']['golden_labels_base'] * 100
    icm = results['results']['icm_base'] * 100

    # Optionally load fixed context results
    if include_fixed and fixed_results_file:
        fixed_results = load_results(fixed_results_file)
        # Check if it's a comparison format or ICM log format
        if 'results' in fixed_results and 'icm_fixed' in fixed_results['results']:
            icm_fixed = fixed_results['results']['icm_fixed'] * 100
        elif 'final' in fixed_results and 'accuracy' in fixed_results['final']:
            # ICM log format - extract accuracy from final section
            icm_fixed = fixed_results['final']['accuracy'] * 100
        else:
            print(f"Warning: Could not find accuracy in fixed results file")
            icm_fixed = None
    else:
        icm_fixed = None

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar positions (ICM third, Golden fourth to match paper Figure 1)
    if include_fixed and icm_fixed is not None:
        x = np.arange(5)
        width = 0.6
        labels = ['Zero-shot', 'Zero-shot\n(Chat)', 'ICM\n(Ours)', 'Golden\nSupervision',
                 'ICM Fixed\nContext']
        accuracies = [zero_shot_base, zero_shot_instruct, icm, golden, icm_fixed]
        colors = [COLORS['zero_shot'], COLORS['zero_shot_chat'], COLORS['unsupervised'],
                 COLORS['golden'], COLORS['icm_fixed']]
    else:
        x = np.arange(4)
        width = 0.6
        labels = ['Zero-shot', 'Zero-shot\n(Chat)', 'ICM\n(Ours)', 'Golden\nSupervision']
        accuracies = [zero_shot_base, zero_shot_instruct, icm, golden]
        colors = [COLORS['zero_shot'], COLORS['zero_shot_chat'], COLORS['unsupervised'],
                 COLORS['golden']]

    # Create bars
    bars = ax.bar(x, accuracies, width, color=colors, edgecolor='black', linewidth=1.2)

    # Add pattern to zero-shot (Chat) bar
    bars[1].set_hatch('o' * 4)

    # Styling
    ax.set_ylabel('accuracy (%)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim([0, max(accuracies) * 1.15])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on top of bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.1f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add title
    dataset_name = results.get('config', {}).get('split', 'TruthfulQA')
    ax.set_title(f'Reproduction of Fig 1 in Feng et al. (2025)\nwith custom ICM implementation - {dataset_name.title()} Split',
                fontsize=12, pad=20)

    # Auto-generate output file name if not specified
    if output_file is None:
        if include_fixed and icm_fixed is not None:
            output_file = 'results/comparison_plot_with_fixed.png'
        else:
            output_file = 'results/comparison_plot.png'

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot comparison results')
    parser.add_argument('results_file', type=str, help='Path to comparison results JSON file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path for plot (auto-generated if not specified)')
    parser.add_argument('--include-fixed', action='store_true',
                       help='Include fixed-context ICM as 5th bar')
    parser.add_argument('--fixed-results', type=str, default=None,
                       help='Path to fixed-context results JSON (required if --include-fixed)')

    args = parser.parse_args()

    plot_comparison(args.results_file, args.output, args.include_fixed, args.fixed_results)
