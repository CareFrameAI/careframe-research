import io
import matplotlib.pyplot as plt
import seaborn as sns

def fig_to_svg(fig):
    # Set figure and axes backgrounds to transparent
    fig.patch.set_alpha(0.0)
    for ax in fig.get_axes():
        ax.patch.set_alpha(0.0)
        # Make legend background transparent if it exists
        if ax.get_legend() is not None:
            ax.get_legend().get_frame().set_alpha(0.0)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', transparent=True)
    buf.seek(0)
    svg_string = buf.getvalue().decode('utf-8')
    buf.close()
    plt.close(fig)
    return svg_string

# Define pastel color palette
PASTEL_COLORS = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFB3F7', '#B3FFF7']
PASTEL_CMAP = sns.color_palette(PASTEL_COLORS)
