#!/bin/bash
# create_figure.sh — Compile, validate, and iterate on a TikZ figure
# Usage: ./create_figure.sh <file.tex> [--max-iter N]
#
# Compiles the figure, runs validation, reports results.
# Designed to be called in a loop by the figure creation agent.

set -euo pipefail

TEX_FILE="${1:?Usage: create_figure.sh <file.tex>}"
MAX_ITER="${2:-1}"
BASE="${TEX_FILE%.tex}"
PDF="${BASE}.pdf"
LOG="${BASE}.log"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Compile ────────────────────────────────────────────────
echo ">>> Compiling ${TEX_FILE}..."
if ! pdflatex -interaction=nonstopmode -halt-on-error "${TEX_FILE}" > /dev/null 2>&1; then
    echo "COMPILE_ERROR"
    echo "---"
    grep -A 3 "^!" "${LOG}" 2>/dev/null || echo "(no ! lines)"
    grep "Undefined control sequence" "${LOG}" 2>/dev/null || true
    grep "File .* not found" "${LOG}" 2>/dev/null || true
    echo "---"
    exit 1
fi
echo "COMPILE_OK: ${PDF}"

# ─── Validate ───────────────────────────────────────────────
echo ""
echo ">>> Validation..."
python "${DIR}/validate_figure.py" "${PDF}" --strict 2>&1 || true

# ─── Font audit ─────────────────────────────────────────────
echo ""
echo ">>> Font size audit:"
grep -oE '\\(tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)' \
    "${TEX_FILE}" 2>/dev/null | sort | uniq -c | sort -rn || echo "  (none found)"

# ─── Overlap heuristic ──────────────────────────────────────
echo ""
echo ">>> Spacing audit (gaps < 6pt):"
python3 -c "
import re, sys
content = open('${TEX_FILE}', encoding='utf-8').read()
spacings = re.findall(r'(above|below|left|right)\s*=\s*([\d.]+)\s*(cm|pt|mm)', content)
tight = []
for d,v,u in spacings:
    val = float(v)
    if u == 'cm': val *= 28.35
    elif u == 'mm': val *= 2.835
    if val < 6: tight.append(f'  {d}={v}{u} (~{val:.1f}pt)')
if tight:
    print(f'{len(tight)} tight spacing(s):')
    for t in tight: print(t)
else:
    print('  All spacings OK (>= 6pt)')
" 2>/dev/null || echo "  (audit skipped)"

# ─── Overlap check: absolute positions ──────────────────────
echo ""
echo ">>> Absolute positioning check:"
ABS=$(grep -cE 'at\s*\([0-9.-]+\s*,\s*[0-9.-]+\)' "${TEX_FILE}" 2>/dev/null || echo 0)
CALC=$(grep -cE 'at\s*\(\$' "${TEX_FILE}" 2>/dev/null || echo 0)
RISKY=$((ABS - CALC))
if [ "$RISKY" -gt 0 ]; then
    echo "  WARNING: ${RISKY} raw absolute placements (prefer positioning library)"
else
    echo "  OK: all positions relative or calc-derived"
fi

# ─── Summary ────────────────────────────────────────────────
echo ""
echo ">>> PDF dimensions:"
python3 -c "
try:
    from PyPDF2 import PdfReader
    r = PdfReader('${PDF}')
    b = r.pages[0].mediabox
    w, h = float(b.width)/72, float(b.height)/72
    print(f'  {w:.2f}\" x {h:.2f}\"')
    if w > 5.7: print('  WARNING: wider than NeurIPS column (5.5\")')
except: print('  (PyPDF2 not available)')
" 2>/dev/null || echo "  (dimension check skipped)"

echo ""
echo "=== DONE: ${PDF} ==="
