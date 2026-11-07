"""Figure implementations for the thesis."""


def save_fig(fig, path_pdf: str, dpi_png: int = 200):
    """Save a figure as PDF and PNG side by side."""
    fig.savefig(path_pdf, dpi=150, bbox_inches="tight")
    path_png = path_pdf.replace(".pdf", ".png")
    fig.savefig(path_png, dpi=dpi_png, bbox_inches="tight")
    print(f"  Saved: {path_pdf}")
    print(f"  Saved: {path_png}")
