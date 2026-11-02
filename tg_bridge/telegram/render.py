"""md_to_html, _md_units, answer chunks (pure, C08).

Pure functions only: markdown in, Telegram HTML or markdown chunks out. No I/O, no clock, no module state beyond the
constants and compiled regexes below. Moved verbatim from tg-bridge.py (C08).
"""

import html
import re

ANSWER_CHUNK = 4000  # HTML characters per answer message (Telegram counts at most 4096 characters after parsing)
ANSWER_MAX_MESSAGES = 3

_FENCE_BLOCK_RE = re.compile(r'(^[ \t]*`{3,}[^\n`]*\n.*?^[ \t]*`{3,}[ \t]*$)', re.S | re.M)  # a fence starts and ends on its own line
_CODE_SPAN_RE = re.compile(r'`{1,3}([^`\n]+?)`{1,3}')
_HEADER_RE = re.compile(r'^#{1,6}[ \t]+(.+?)[ \t]*$', re.M)
_BOLD_RE = re.compile(r'\*\*(?=\S)(.+?)(?<=\S)\*\*')
_ITALIC_STAR_RE = re.compile(r'(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])')
_ITALIC_UNDER_RE = re.compile(r'(?<!\w)_(?=\S)([^_\n]+?)(?<=\S)_(?!\w)')
_LINK_RE = re.compile(r'\[([^\]\n]+)\]\((https?://[^\s)"<>]+)\)')
_TABLE_SEP_RE = re.compile(r'^\|?[ \t]*:?-+:?[ \t]*(\|[ \t]*:?-+:?[ \t]*)*\|?$')  # the |---|:--:| row under a table header
_CELL_TOKEN_RE = re.compile(r'(`+).*?(?<!`)\1(?!`)|\\\||\|')  # code spans and escaped pipes are skipped, a bare | splits


def _split_cells(s):
    """Split a row body on its bare pipes only: a pipe inside a backtick code span or after a backslash is text."""
    cells, last = [], 0
    for m in _CELL_TOKEN_RE.finditer(s):
        if m.group(0) == '|':
            cells.append(s[last:m.start()])
            last = m.end()
    cells.append(s[last:])
    return cells


def _table_cells(line):
    """The cells of one table row: the outer pipes dropped, each cell stripped, and an escaped \\| turned into a
    literal |."""
    s = line.strip()
    s = s[1:] if s.startswith('|') else s
    s = s[:-1] if s.endswith('|') and not s.endswith('\\|') else s
    return [c.strip().replace('\\|', '|') for c in _split_cells(s)]


def _table_row_line(header, cells):
    """One body row as a bullet: '• Header: cell · Header: cell'. Empty cells are skipped; a column without a header
    label, or a one-column table, shows its cell alone. None when the row has no cell text at all."""
    labelled = len(header) > 1
    pairs = []
    for idx, cell in enumerate(cells):
        if not cell:
            continue
        label = header[idx] if labelled and idx < len(header) else ''
        pairs.append(f'{label}: {cell}' if label else cell)
    return '• ' + ' · '.join(pairs) if pairs else None


def _tables_as_lists(seg):
    """Markdown tables in one stretch of prose become bullet lines (Telegram cannot show tables). A table is a header
    row '| … |', a separator row of dashes, then body rows containing a pipe; any other line is kept as it is."""
    lines = seg.split('\n')
    out, i = [], 0
    while i < len(lines):
        head = lines[i].strip()
        sep = lines[i + 1].strip() if i + 1 < len(lines) else ''
        is_table = (len(head) > 1 and head.startswith('|') and head.endswith('|')
                    and '|' in sep and _TABLE_SEP_RE.match(sep))
        if not is_table:
            out.append(lines[i])
            i += 1
            continue
        header = _table_cells(head)
        j, start = i + 2, len(out)
        while j < len(lines) and lines[j].strip() and '|' in lines[j]:
            body = lines[j].strip()
            row = None if _TABLE_SEP_RE.match(body) else _table_row_line(header, _table_cells(body))  # a repeated |---| row is dropped
            if row:
                out.append(row)
            j += 1
        if len(out) == start:  # no body row to label: the header cells alone, as one line
            row = _table_row_line([], header)
            if row:
                out.append(row)
        i = j
    return '\n'.join(out)


def _tables_outside_fences(text):
    """The whole markdown with its tables already turned into bullet lines; fenced blocks are returned untouched."""
    parts = _FENCE_BLOCK_RE.split(text)
    return ''.join(p if i % 2 else _tables_as_lists(p) for i, p in enumerate(parts))


def _prose_html(seg):
    """One stretch of prose (no fenced block) to HTML. Inline code is split out first, so nothing inside a code
    span is formatted; links come last, so no other rule can touch a URL."""
    seg = _HEADER_RE.sub(r'<b>\1</b>', seg)
    out = []
    for i, piece in enumerate(_CODE_SPAN_RE.split(seg)):
        if i % 2:
            out.append(f'<code>{piece}</code>')
            continue
        piece = _BOLD_RE.sub(r'<b>\1</b>', piece)
        piece = _ITALIC_STAR_RE.sub(r'<i>\1</i>', piece)
        piece = _ITALIC_UNDER_RE.sub(r'<i>\1</i>', piece)
        out.append(_LINK_RE.sub(r'<a href="\2">\1</a>', piece))
    return ''.join(out)


def _fence_body(part):
    """The code inside one fenced block: its opening line (with the language) and its closing fence line removed."""
    return '\n'.join(part.split('\n')[1:-1])


def md_to_html(text):
    """Telegram HTML from the model's markdown. &, < and > are escaped first, so the output is always well-formed.
    Fenced blocks become <pre><code>; the rest goes through _prose_html."""
    out = []
    for i, part in enumerate(_FENCE_BLOCK_RE.split(html.escape(text or '', quote=False))):
        if i % 2:  # a fenced block
            out.append(f'<pre><code>{_fence_body(part).strip(chr(10))}</code></pre>')
        else:
            out.append(_prose_html(_tables_as_lists(part)))
    return ''.join(out)


def _md_units(text):
    """The markdown as (text, is_code) units that never split a fenced block: each fence is one unit, prose is
    split into paragraphs at blank lines."""
    units = []
    for i, part in enumerate(_FENCE_BLOCK_RE.split(text)):
        if i % 2:
            units.append((part, True))
        else:
            units += [(p.strip('\n'), False) for p in re.split(r'\n[ \t]*\n', part) if p.strip()]
    return units


def _fit(md, is_code):
    """md itself when its HTML fits one message; otherwise pieces cut at line boundaries (a code block is re-fenced
    for every piece, so each piece is a complete <pre> block)."""
    if len(md_to_html(md)) <= ANSWER_CHUNK:
        return [md]
    head = md.partition('\n')[0] if is_code else ''
    if is_code:
        lines = _fence_body(md).strip('\n').split('\n')
    else:
        lines = md.split('\n')

    def wrap(body):
        return f'{head}\n{body}\n```' if is_code else body
    pieces, cur = [], []
    for ln in lines:
        if cur and len(md_to_html(wrap('\n'.join(cur + [ln])))) > ANSWER_CHUNK:
            pieces.append(wrap('\n'.join(cur)))
            cur = []
        cur.append(ln)
    pieces.append(wrap('\n'.join(cur)))
    return pieces


def _answer_chunks(text):
    """The answer as markdown chunks that each fit one message, split only between units. None when it would
    need more than ANSWER_MAX_MESSAGES messages, or a unit cannot fit at all."""
    chunks, cur = [], ''
    for md, is_code in _md_units(_tables_outside_fences(text)):  # tables become bullets before any cut, so no chunk gets a bare header-less row
        for piece in _fit(md, is_code):
            cand = f'{cur}\n\n{piece}' if cur else piece
            if len(md_to_html(cand)) <= ANSWER_CHUNK:
                cur = cand
                continue
            if cur:
                chunks.append(cur)
            cur = piece
    if cur:
        chunks.append(cur)
    if len(chunks) > ANSWER_MAX_MESSAGES or any(len(md_to_html(c)) > ANSWER_CHUNK for c in chunks):
        return None
    return chunks
