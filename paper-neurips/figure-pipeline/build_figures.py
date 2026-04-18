#!/usr/bin/env python3
"""Build all TikZ figures and copy to ../figures/"""
import subprocess, shutil, os

PIPE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(PIPE_DIR), "figures")

tex_files = [
    "fig_architecture_sigreg.tex",
]

for tex in tex_files:
    tex_path = os.path.join(PIPE_DIR, tex)
    print(f"Compiling {tex}...")
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex],
        cwd=PIPE_DIR, capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stdout[-200:] if result.stdout else 'no output'}")
    else:
        print(f"  OK")

# Copy all PDFs
pdfs = [
    "fig_domain_overview.pdf",
    "fig_tokenization.pdf",
    "fig_architecture_ema.pdf",
    "fig_architecture_sigreg.pdf",
]

for pdf in pdfs:
    src = os.path.join(PIPE_DIR, pdf)
    dst = os.path.join(FIG_DIR, pdf)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {pdf} -> figures/")
    else:
        print(f"MISSING: {src}")

print("Done.")
