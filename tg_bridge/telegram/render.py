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
            out.append(_prose_html(part))
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
    for md, is_code in _md_units(text):
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
