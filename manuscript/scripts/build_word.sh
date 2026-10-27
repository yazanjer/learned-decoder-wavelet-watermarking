#!/usr/bin/env bash
# V1.0 Word build (run from manuscript/ after `latexmk -pdf main.tex supplementary.tex`, which writes the .aux files).
# Flattens \input, swaps figure PDFs for the 300 dpi PNGs, resolves \ref/\eqref from the .aux files,
# and converts with pandoc (citeproc, references.bib) and scripts/simple_math.lua.
set -euo pipefail
TMP=$(mktemp -d)
python3 - "$TMP" <<'PY'
import re, sys
T = sys.argv[1]
def flat(p):
    t = open(p).read()
    return re.sub(r'\\input\{([^}]+)\}', lambda m: flat(m.group(1) if m.group(1).endswith('.tex') else m.group(1) + '.tex'), t)
for n in ('main', 'supplementary'):
    t = flat(f'{n}.tex')
    t = re.sub(r'(figures/[A-Za-z0-9_]+)\.pdf', r'\1.png', t).replace('\\linenumbers', '')
    t = re.sub(r'\\(Before|After)(Begin|End)Environment\{tabular\}\{[^\n]*\n', '', t)
    t = t.replace('\\begin{frontmatter}', '').replace('\\end{frontmatter}', '')
    t = re.sub(r'\\begin\{keyword\}(.*?)\\end\{keyword\}', lambda m: '\\paragraph{Keywords} ' + m.group(1).replace('\\sep', ';'), t, flags=re.S)
    for a, b in (('\\Bigl(', '\\left('), ('\\Bigr)', '\\right)'), ('\\bigl(', '\\left('), ('\\bigr)', '\\right)')):
        t = t.replace(a, b)
    lab = {m.group(1): m.group(2) for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}', open(f'{n}.aux').read())}
    t = re.sub(r'\\eqref\{([^}]+)\}', lambda m: '(' + lab.get(m.group(1), '??') + ')', t)
    t = re.sub(r'\\ref\{([^}]+)\}', lambda m: lab.get(m.group(1), '??'), t)
    if n == 'supplementary':
        t = t.replace('\\renewcommand{\\thetable}{S\\arabic{table}}', '')
    open(f'{T}/{n}.tex', 'w').write(t)
PY
VER=${1:-V1.0}
mkdir -p word
for n in main supplementary; do
  pandoc "$TMP/$n.tex" -f latex -t docx --citeproc --bibliography=references.bib --resource-path=. \
    --lua-filter=scripts/simple_math.lua -o "word/${n}_${VER}.docx"
done
rm -rf "$TMP"
