"""Lossless, self-contained SVG/PDF wrappers for real-frame composites."""

import base64
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
try:
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
except ModuleNotFoundError:
    ImageReader = None
    canvas = None


ROOT = Path(__file__).resolve().parent / 'generated'
FIGURES = {
    "vehicle_before_endpoints": "Before learning: real video frame composite",
    "vehicle_after_endpoints": "After learning: real video frame composite",
    "vehicle_before_after_endpoints": "Before and after learning: real video frame composites",
}


def main():
    pdf_dir = ROOT
    svg_dir = ROOT
    pdf_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    for stem, title in FIGURES.items():
        source = ROOT / f"{stem}.png"
        with Image.open(source) as raster:
            width, height = raster.size
        # Preserve every original pixel; no tracing, resizing, or JPEG encoding.
        encoded = base64.b64encode(source.read_bytes()).decode("ascii")
        svg = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{width}px" height="{height}px" viewBox="0 0 {width} {height}">\n'
            f'  <title>{escape(title)}</title>\n'
            '  <desc>Real video pixels, alpha composited and annotated by script. '
            'The embedded image is raster, not vector-traced.</desc>\n'
            f'  <image x="0" y="0" width="{width}" height="{height}" '
            f'xlink:href="data:image/png;base64,{encoded}"/>\n'
            '</svg>\n'
        )
        (svg_dir / f"{stem}.svg").write_text(svg, encoding="utf-8")

        # Nominal publication size; aspect ratio is exact, with no page margins.
        page_width = 516.0 if "before_after" in stem else 252.0
        page_height = page_width * height / width
        pdf_path = pdf_dir / f"{stem}.pdf"
        if canvas is not None:
            pdf = canvas.Canvas(
                str(pdf_path),
                pagesize=(page_width, page_height),
                pageCompression=1,
            )
            pdf.setTitle(title)
            pdf.setSubject("Lossless embedded real-video composite; no vector tracing")
            pdf.drawImage(ImageReader(str(source)), 0, 0, page_width, page_height)
            pdf.showPage()
            pdf.save()
        else:
            # Pillow fallback preserves the full-resolution raster and the
            # same nominal publication width when reportlab is unavailable.
            resolution = width * 72.0 / page_width
            with Image.open(source) as raster:
                raster.convert("RGB").save(pdf_path, "PDF", resolution=resolution)
        print(f"Exported {stem}: {width} x {height}, PDF and SVG")


if __name__ == "__main__":
    main()
