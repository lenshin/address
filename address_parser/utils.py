import json
from itertools import cycle

from ipymarkup import show_box_markup
from ipymarkup.palette import PALETTE, BLUE, RED, GREEN, PURPLE, BROWN, ORANGE

from address_parser.config import LABEL_PATH

with open(LABEL_PATH, 'r') as file:
    label_dict = json.load(file)
ADDRESS_MASK = label_dict.keys()

cl = cycle([BLUE, RED, GREEN, PURPLE, BROWN, ORANGE])
COLORS = [next(cl) for _ in range(len(ADDRESS_MASK))]

for tag, color in zip(ADDRESS_MASK, COLORS):
    PALETTE.set(tag, color)


def show_predict(tokens, tags):
    def mapper(tag):
        return tag[2:] if tag != 'OTHER' else tag

    tags = [mapper(tag) for tag in tags]
    text = ' '.join(tokens)
    spans = []

    start, end, tag = 0, len(tokens[0]), tags[0]

    for word, ttag in zip(tokens[1:], tags[1:]):
        if tag == ttag:
            end += 1 + len(word)
        else:
            span = (start, end, tag)
            spans.append(span)

            start = 1 + end
            end += 1 + len(word)
            tag = ttag

    span = (start, end, tag)
    spans.append(span)
    spans = filter(lambda x: x[2] not in {'other', 'comma'}, spans)
    show_box_markup(text, spans, palette=PALETTE)
