"""Build the single-file submission (docs/SPEC.md A6): concatenate src/{config,data,text,models,
portfolio,evaluate}.py and MAIN.py's body into outputs/submission/MAIN.py, needing no src/
package. Each module's stripped source is embedded verbatim as an `r'''<source>'''` raw string
literal (under a `# ===== src/x.py =====` banner, so a judge can read the whole bundle top to
bottom without any repr()-escaping noise) and exec'd into its own module object, pre-seeded with
the dependency modules/names it references, so `config.X`/`data.f` etc. resolve with no
identifier rewriting, and `if __name__=='__main__'` guards never fire.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'submission' / 'MAIN.py'
_IMPORT_RE = re.compile(r'^[ \t]*(from src(\.\w+)? import [^\n]*|import src(\.\w+)?)[ \t]*$', re.MULTILINE)

def _stripped(path):
    return _IMPORT_RE.sub('', path.read_text(encoding='utf-8'))

def _as_raw_literal(name, src):
    """Embed `src` as an r'''...''' literal. Guards the two ways that could silently corrupt the
    bundle: a literal ''' inside the source would close the string early, and a source ending in
    an odd run of backslashes would escape the closing quote (raw strings still can't end that
    way -- the backslash-quote pairing is a lexer rule, not a raw-string exemption)."""
    assert "'''" not in src, f"{name}: source contains ''' -- cannot embed as an r'''...''' literal"
    trailing_bs = len(src) - len(src.rstrip('\\'))
    assert trailing_bs % 2 == 0, (
        f"{name}: source ends with an odd run of backslashes -- cannot embed as a raw string")
    return f"r'''{src}'''"

def _config_source():
    # bundled elsewhere: ROOT <- env var ALPHABERT_ROOT, fallback cwd (not __file__-relative).
    old = "ROOT = Path(__file__).resolve().parents[1]"
    new = "import os as _os\nROOT = Path(_os.environ.get('ALPHABERT_ROOT', '.')).resolve()"
    src = _stripped(ROOT / 'src' / 'config.py')
    assert old in src, 'config.py ROOT line not found'
    return src.replace(old, new)

HEADER = '''"""AlphaBERT -- single-file competition submission (docs/SPEC.md A6).
Bundled by build_submission_main.py; needs no src/ package. Paths (data/, outputs/) resolve
relative to env var ALPHABERT_ROOT (fallback: cwd) -- set it when running from elsewhere:
    ALPHABERT_ROOT=/path/to/AlphaBERT python MAIN.py --dry
"""
import sys, types

def _load_module(name, source, **deps):
    mod = types.ModuleType(name)
    mod.__dict__.update(deps)
    exec(compile(source, f"<bundled:{name}>", "exec"), mod.__dict__)
    sys.modules[name] = mod
    return mod
'''

def build():
    src = ROOT / 'src'
    m = {n: _stripped(src / f'{n}.py') for n in ('data', 'text', 'models', 'portfolio', 'evaluate')}
    m['config'] = _config_source()

    blocks = [
        "# ===== src/config.py =====\n"
        f"config = _load_module('config', {_as_raw_literal('config', m['config'])})\n",

        "# ===== src/data.py =====\n"
        f"data = _load_module('data', {_as_raw_literal('data', m['data'])}, config=config)\n",

        "# ===== src/text.py =====\n"
        f"text = _load_module('text', {_as_raw_literal('text', m['text'])}, config=config, data=data)\n",

        "# ===== src/models.py =====\n"
        f"models = _load_module('models', {_as_raw_literal('models', m['models'])}, config=config,\n"
        f"    feature_columns=data.feature_columns, TEXT_FEATURES=text.TEXT_FEATURES)\n",

        "# ===== src/portfolio.py =====\n"
        f"portfolio = _load_module('portfolio', {_as_raw_literal('portfolio', m['portfolio'])}, config=config)\n",

        "# ===== src/evaluate.py =====\n"
        f"evaluate = _load_module('evaluate', {_as_raw_literal('evaluate', m['evaluate'])}, config=config)\n",
    ]
    calls = '\n'.join(blocks)
    out = (HEADER + '\n' + calls + '\n# ===== MAIN.py =====\n'
           + _stripped(ROOT / 'MAIN.py'))  # MAIN's own __main__ guard fires as-is
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(out, encoding='utf-8')
    print(f'build_submission_main: wrote {OUT} ({len(out)} chars, {out.count(chr(10))} lines)')

if __name__ == '__main__':
    build()
