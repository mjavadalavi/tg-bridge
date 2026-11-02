"""render: port of the pure-function checks of fakecli/test_answer.py (labels A, B, D, F, J, K kept in docstrings).

Checks C, E, G, H and I of that suite send messages or documents and are ported with telegram/answers.py. Nothing here
touches the network, the clock or the filesystem.
"""

import unittest

from tg_bridge.telegram.render import (
    ANSWER_CHUNK,
    ANSWER_MAX_MESSAGES,
    _answer_chunks,
    _md_units,
    md_to_html,
)


def _pre_balanced(text: str) -> bool:
    return text.count('<pre>') == text.count('</pre>')


class ConstantsTests(unittest.TestCase):
    """The answer limits the old file used (4000 HTML characters, three messages)."""

    def test_limits(self) -> None:
        self.assertEqual(ANSWER_CHUNK, 4000)
        self.assertEqual(ANSWER_MAX_MESSAGES, 3)


class EscapingAndInlineTests(unittest.TestCase):
    """A) escaping and inline rules."""

    def test_amp_lt_gt_escaped_first(self) -> None:
        """A: &, < and > escaped first"""
        self.assertEqual(md_to_html('a < b & c > d'), 'a &lt; b &amp; c &gt; d')

    def test_header_becomes_bold_line(self) -> None:
        """A: header becomes bold line"""
        self.assertEqual(md_to_html('## Title'), '<b>Title</b>')

    def test_bold_and_italic(self) -> None:
        """A: bold and italic"""
        self.assertEqual(md_to_html('**bold** and *it* and _u_'), '<b>bold</b> and <i>it</i> and <i>u</i>')

    def test_inline_code_is_not_formatted(self) -> None:
        """A: inline code is not formatted"""
        self.assertEqual(md_to_html('use `a_b*c*` now and **x**'), 'use <code>a_b*c*</code> now and <b>x</b>')

    def test_snake_case_paths_and_dunders_untouched(self) -> None:
        """A: snake_case, paths and dunders untouched"""
        text = 'my_var_name and /home/x/some_file_name.py and __init__'
        self.assertEqual(md_to_html(text), text)

    def test_link_with_escaped_url(self) -> None:
        """A: link with escaped URL"""
        self.assertEqual(
            md_to_html('see [docs](https://ex.com/a?x=1&y=2)'),
            'see <a href="https://ex.com/a?x=1&amp;y=2">docs</a>',
        )

    def test_empty_and_none_input(self) -> None:
        self.assertEqual(md_to_html(''), '')
        self.assertEqual(md_to_html(None), '')


class FencedCodeTests(unittest.TestCase):
    """B) fenced code: escaped, never formatted inside."""

    def test_fenced_block_escaped_in_pre_code(self) -> None:
        """B: fenced block escaped in <pre><code>"""
        self.assertEqual(
            md_to_html('x\n```py\nprint("a<b")\n```\ny'),
            'x\n<pre><code>print("a&lt;b")</code></pre>\ny',
        )

    def test_nothing_formatted_inside_a_fence(self) -> None:
        """B: nothing formatted inside a fence"""
        self.assertEqual(md_to_html('```\n**not bold** `x`\n```'), '<pre><code>**not bold** `x`</code></pre>')


class InlineTripleBacktickTests(unittest.TestCase):
    """J) an inline ``` span is not a fence: its words survive and the real fence after it still works."""

    def test_inline_triple_backticks_are_code_and_later_fence_intact(self) -> None:
        """J: inline triple backticks are code, the later fence is intact"""
        self.assertEqual(
            md_to_html('Use ```x``` here\nand\n```py\ncode\n```\ntail'),
            'Use <code>x</code> here\nand\n<pre><code>code</code></pre>\ntail',
        )

    def test_inline_triple_backtick_span_rendered_as_code(self) -> None:
        """J: inline triple-backtick span rendered as code"""
        self.assertEqual(md_to_html('a ```b``` c'), 'a <code>b</code> c')

    def test_markdown_units_keep_inline_span_in_prose(self) -> None:
        """J: markdown units keep the inline span in prose"""
        self.assertEqual(
            _md_units('Use ```x``` here\nand\n```py\ncode\n```\ntail'),
            [('Use ```x``` here\nand', False), ('```py\ncode\n```', True), ('tail', False)],
        )


class FourBacktickFenceTests(unittest.TestCase):
    """K) a 4-backtick fence: no stray backtick, no empty <pre>."""

    def test_four_backtick_fence_becomes_one_clean_block(self) -> None:
        """K: 4-backtick fence becomes one clean block"""
        self.assertEqual(md_to_html('````py\ncode\n````\ntail'), '<pre><code>code</code></pre>\ntail')

    def test_empty_fenced_block(self) -> None:
        """K: empty fenced block"""
        self.assertEqual(md_to_html('```py\n```'), '<pre><code></code></pre>')


class AnswerChunkTests(unittest.TestCase):
    """D, F and K (split part): chunking of long answers, markdown only; sending is tested in test_answers."""

    def test_long_answer_splits_into_two_or_three_chunks_with_code_whole(self) -> None:
        """D: split into 2-3 messages"""
        code = '```py\n' + '\n'.join(f'value_{i} = {i}  # note' for i in range(260)) + '\n```'
        long_ans = 'intro paragraph\n\n' + code + '\n\n' + ('filler text ' * 120) + '\n\noutro'
        chunks = _answer_chunks(long_ans)
        self.assertIsNotNone(chunks)
        self.assertTrue(1 < len(chunks) <= 3, f'got {len(chunks)} chunks')
        html_parts = [md_to_html(c) for c in chunks]
        self.assertTrue(all(len(h) <= 4000 for h in html_parts), 'every message within 4000 characters')
        self.assertTrue(all(_pre_balanced(h) for h in html_parts), 'no code block split across messages')

    def test_four_chunks_worth_of_text_does_not_fit(self) -> None:
        """F: does not fit three messages -> document instead of truncation"""
        four = '\n\n'.join(f'para {i} ' + 'w' * 2100 for i in range(4))  # 8400 characters, each paragraph needs its own message
        self.assertIsNone(_answer_chunks(four))

    def test_unit_that_cannot_fit_at_all_gives_none(self) -> None:
        """F: a single unit larger than one message gives None, nothing is cut"""
        self.assertIsNone(_answer_chunks('w' * 9000))

    def test_short_answer_is_one_chunk_with_its_text(self) -> None:
        """C (text part): the converted answer is the one chunk"""
        text = 'First **para**.\n\nSecond para with `code`.'
        chunks = _answer_chunks(text)
        self.assertEqual(chunks, [text])
        self.assertEqual(md_to_html(chunks[0]), 'First <b>para</b>.\n\nSecond para with <code>code</code>.')

    def test_four_backtick_block_split_over_chunks(self) -> None:
        """K: long 4-backtick block split over messages"""
        code4 = '````py\n' + '\n'.join(f'value_{i} = {i}' for i in range(500)) + '\n````'
        chunks = _answer_chunks(code4)
        self.assertIsNotNone(chunks)
        self.assertTrue(1 < len(chunks) <= 3, f'got {len(chunks)} chunks')
        html_parts = [md_to_html(c) for c in chunks]
        self.assertTrue(all('`' not in h and len(h) <= 4000 for h in html_parts), 'no stray backtick in any piece')
        self.assertIn('value_499 = 499', ''.join(html_parts), 'the last line of the block is kept')


class MarkdownUnitTests(unittest.TestCase):
    """_md_units: fences are whole units, prose splits at blank lines."""

    def test_prose_paragraphs_and_fence_units(self) -> None:
        self.assertEqual(
            _md_units('one\n\ntwo\n```\nc\n```\nthree'),
            [('one', False), ('two', False), ('```\nc\n```', True), ('three', False)],
        )

    def test_blank_only_input_gives_no_units(self) -> None:
        self.assertEqual(_md_units('   \n\n  '), [])


class MarkdownTableTests(unittest.TestCase):
    """Markdown tables in prose become bullet lines with header labels; tables inside fences and lone pipes stay."""

    def test_three_column_table_becomes_bullets_with_labels_and_empty_cells_skipped(self) -> None:
        text = "| Name | Price | Stock |\n|---|---|---|\n| Tea | 10 | 3 |\n| Milk |  | 5 |"
        self.assertEqual(md_to_html(text), "• Name: Tea · Price: 10 · Stock: 3\n• Name: Milk · Stock: 5")

    def test_table_inside_fence_is_untouched(self) -> None:
        text = "```\n| a | b |\n|---|---|\n| 1 | 2 |\n```"
        self.assertEqual(md_to_html(text), "<pre><code>| a | b |\n|---|---|\n| 1 | 2 |</code></pre>")

    def test_persian_table_keeps_its_text(self) -> None:
        text = "| نام | قیمت |\n|---|---|\n| چای | ۱۰ هزار تومان |"
        self.assertEqual(md_to_html(text), "• نام: چای · قیمت: ۱۰ هزار تومان")

    def test_text_before_and_after_the_table_is_kept(self) -> None:
        text = "Before\n| a | b |\n|---|---|\n| 1 | 2 |\nAfter"
        self.assertEqual(md_to_html(text), "Before\n• a: 1 · b: 2\nAfter")

    def test_inline_markdown_in_cells_is_converted(self) -> None:
        text = "| **Item** | `Code` |\n|---|---|\n| [site](https://e.io) | _z_ |"
        self.assertEqual(md_to_html(text),
                         '• <b>Item</b>: <a href="https://e.io">site</a> · <code>Code</code>: <i>z</i>')

    def test_empty_header_gives_plain_cells(self) -> None:
        text = "|  |  |\n|---|---|\n| a | b |"
        self.assertEqual(md_to_html(text), "• a · b")

    def test_pipe_inside_code_span_does_not_split_the_cell(self) -> None:
        text = "| a | b |\n|---|---|\n| `x|y` | 2 |"
        self.assertEqual(md_to_html(text), "• a: <code>x|y</code> · b: 2")

    def test_escaped_pipe_becomes_literal_pipe_in_cell(self) -> None:
        text = "| a | b |\n|---|---|\n| p\\|q | 2 |"
        self.assertEqual(md_to_html(text), "• a: p|q · b: 2")

    def test_header_without_body_rows_is_kept_as_one_line(self) -> None:
        self.assertEqual(md_to_html("| a | b |\n|---|---|"), "• a · b")

    def test_repeated_separator_row_in_body_is_skipped(self) -> None:
        text = "| a | b |\n|---|---|\n| 1 | 2 |\n|---|---|\n| 3 | 4 |"
        self.assertEqual(md_to_html(text), "• a: 1 · b: 2\n• a: 3 · b: 4")

    def test_long_table_split_into_chunks_has_no_raw_table_rows(self) -> None:
        table = "| name | val |\n|---|---|\n" + "\n".join(f"| r{i} | v |" for i in range(400))
        chunks = _answer_chunks(table)
        self.assertIsNotNone(chunks)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertFalse(any(line.startswith('|') for line in md_to_html(chunk).split('\n')))
            self.assertTrue(all(line.startswith('•') for line in md_to_html(chunk).split('\n')))
        self.assertIn('• name: r399 · val: v', '\n'.join(md_to_html(c) for c in chunks))

    def test_lone_pipe_line_is_unchanged(self) -> None:
        self.assertEqual(md_to_html("a | b"), "a | b")


if __name__ == '__main__':
    unittest.main()
