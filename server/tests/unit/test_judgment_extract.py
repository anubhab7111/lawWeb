from app.ingest.judgment_extract import pages_to_paragraphs, parse_bbox_xml


def line(x0, y0, text, x1=None, h=10):
    x1 = x1 if x1 is not None else x0 + 7 * len(text)
    words = "".join(f'<word xMin="{x0}" yMin="{y0}" xMax="{x1}" yMax="{y0 + h}">{w}</word>' for w in text.split())
    return f'<line xMin="{x0}" yMin="{y0}" xMax="{x1}" yMax="{y0 + h}">{words}</line>'


def page(*lines, w=430, h=640):
    return f'<page width="{w}" height="{h}"><flow><block>{"".join(lines)}</block></flow></page>'


BODY = 68  # body left edge; paragraph starts sit ~23pt right of it


def doc(*pages):
    return f"<doc>{''.join(pages)}</doc>"


def body_lines(start_y, texts, first_indent=True):
    out = []
    for i, text in enumerate(texts):
        x0 = BODY + (23 if (i == 0 and first_indent) else 0)
        out.append(line(x0, start_y + 13 * i, text, x1=376))
    return out


def test_margin_letters_headers_and_page_numbers_are_dropped_by_position():
    first = page(*body_lines(80, ["Title page paragraph text is here."]))
    second = page(
        line(165, 56, "PROF. YASHPAL v. STATE", x1=277),
        line(364, 55, "27", x1=375),
        *body_lines(80, ["The Court held that the provision was valid and", "in consequence the appeal was dismissed."]),
        line(383, 80, "A", x1=392),
        line(383, 93, "B", x1=392),
    )
    text = "\n\n".join(pages_to_paragraphs(parse_bbox_xml(doc(first, second))))
    assert "YASHPAL" not in text and "27" not in text
    assert " A " not in f" {text} " and " B " not in f" {text} "
    assert "valid and in consequence the appeal" in text


def test_indented_lines_start_paragraphs_only_after_a_sentence_end():
    lines = [
        line(BODY + 23, 80, "First paragraph begins here and wraps", x1=376),
        line(BODY, 93, "onto a second line that ends the sentence.", x1=376),
        line(BODY + 23, 106, "Second paragraph starts here as well", x1=376),
        line(BODY, 119, "and continues to the end of the text.", x1=376),
        line(BODY, 132, "Filler line to establish the body edge of text.", x1=376),
        line(BODY, 145, "Another filler line of body text with width here.", x1=376),
    ]
    paragraphs = pages_to_paragraphs(parse_bbox_xml(doc(page(*lines))))
    assert paragraphs[0] == "First paragraph begins here and wraps onto a second line that ends the sentence."
    assert paragraphs[1].startswith("Second paragraph starts here as well and continues")


def test_page_break_mid_sentence_joins_and_hyphenation_is_repaired():
    p1 = page(*body_lines(80, ["Body text line one is here to set the edges", "Body text line two is here to set edges", "Body text line three fills the page too", "The order of the High Court was miscon-"]))
    p2 = page(
        line(165, 56, "RUNNING HEADER 28", x1=277),
        *body_lines(80, ["duct within the meaning of the rules.", "Filler."], first_indent=False),
    )
    text = "\n\n".join(pages_to_paragraphs(parse_bbox_xml(doc(p1, p2))))
    assert "misconduct within the meaning of the rules." in text


def line_words(y0, words, h=10):
    x0, x1 = words[0][0], words[-1][1]
    inner = "".join(f'<word xMin="{a}" yMin="{y0}" xMax="{b}" yMax="{y0 + h}">{t}</word>' for a, b, t in words)
    return f'<line xMin="{x0}" yMin="{y0}" xMax="{x1}" yMax="{y0 + h}">{inner}</line>'


def test_margin_letters_inside_a_body_line_are_removed_per_word():
    lines = [
        line(BODY, 80, "The appellant relied on the settled view of the Court that", x1=376),
        line(BODY, 93, "the statute must be read in its entirety and in context", x1=376),
        line(BODY, 106, "so that no provision is rendered otiose or superfluous", x1=376),
        line(BODY, 119, "and the object of the enactment is fully served here", x1=376),
        # marker inside the line's word list, left of the body edge
        line_words(132, [(20, 28, "D"), (68, 100, "provisions"), (104, 130, "of"), (134, 160, "the"), (164, 376, "2000 Act were held applicable.")]),
        # marker at the right edge, same line
        line_words(145, [(68, 200, "Learned counsel argued that"), (204, 376, "the notification is void"), (384, 392, "E")]),
    ]
    text = "\n\n".join(pages_to_paragraphs(parse_bbox_xml(doc(page(*lines)))))
    assert "D provisions" not in text and " E" not in text.split("void")[-1]
    assert "provisions of the 2000 Act were held applicable." in text
    assert text.rstrip().endswith("notification is void")
