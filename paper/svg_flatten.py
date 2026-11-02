"""Flatten a one-page PDF into a self-contained SVG (text and embedded symbols as plain paths) for the README.
Usage: python svg_flatten.py in.pdf out.svg"""
import sys, pymupdf

page = pymupdf.open(sys.argv[1])[0]
open(sys.argv[2], "w").write(page.get_svg_image(text_as_path=True))
