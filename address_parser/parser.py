from collections import defaultdict

from config import BUILDING_ENTITY, BUILDING_KEY, COMMA_TAG, UNPARSED_KEY, STREET_KEY, STREET_TYPE_KEY, SEVERAL_STREETS
from config import LEMMA


def bio_tagging_fix(pred_tags):
    start = 0
    while start < len(pred_tags) and pred_tags[start][0] == 'I':
        pred_tags[start] = 'B-other'
        start += 1
    for index in range(start, len(pred_tags) - 1):
        if pred_tags[index - 1][0] == 'I'\
                and (pred_tags[index - 1] == pred_tags[index + 1]) \
                and (pred_tags[index - 1][2:] != pred_tags[index][2:]):
            pred_tags[index] = pred_tags[index - 1]
    return pred_tags


def bio_to_tags(tokens, pred_tags, probas):
    pred_tags = bio_tagging_fix(pred_tags)

    new_pred_tags = [pred_tags[0][2:]]
    new_tokens = [tokens[0]]
    new_probas = [probas[0]]
    prev = pred_tags[0]

    for index in range(1, len(tokens)):
        cur = pred_tags[index]
        if cur[0] == 'I' and prev[2:] == cur[2:]:
            new_tokens[-1] += ' ' + tokens[index]
        else:
            new_tokens.append(tokens[index])
            new_pred_tags.append(pred_tags[index][2:])
            new_probas.append(probas[index])
        prev = cur
    return new_tokens, new_pred_tags, probas


def lemma_type(tokens, pred_tags):
    for index, tag in enumerate(pred_tags):
        if tag in LEMMA.keys():
            tokens[index] = LEMMA.get(tag).get(tokens[index], tokens[index])
    return tokens, pred_tags

def process_tag(tokens, pred_tags, probas):
    answer = defaultdict(list)
    if len(tokens) < 1:
        return answer
    tokens, pred_tags, probas = bio_to_tags(tokens, pred_tags, probas)
    tokens, pred_tags = lemma_type(tokens, pred_tags)

    tag_checked = {COMMA_TAG}
    for index, tag in enumerate(pred_tags):
        if tag in tag_checked:
            if tag != COMMA_TAG:
                answer['other'].append(tokens[index])
        elif tag in {STREET_KEY}:
            answer[tag].append(tokens[index])
        elif tag in BUILDING_ENTITY:
            if not answer.get(BUILDING_KEY) or tag in answer[BUILDING_KEY][-1].keys():
                answer[BUILDING_KEY].append({tag: tokens[index]})
            else:
                answer[BUILDING_KEY][-1][tag] = tokens[index]
        else:
            answer[tag].append(tokens[index])
            tag_checked.add(tag)
    return answer


def multi_street(address_dict):
    if len(address_dict.get(STREET_KEY, [])) > 0:
        address_dict[STREET_TYPE_KEY] = [SEVERAL_STREETS]


def extract_address(address_dict, probability=None):
    """
    Convert address_dict in list of addresses which contain in address string
    :address_dict: dict
    :probability: probability that address parsed good enough
    """
    result = []
    main_adddress = {}
    # multi_street(address_dict)
    for key, value in address_dict.items():
        if key == 'other':
            main_adddress[UNPARSED_KEY] = value
        elif key != BUILDING_KEY:
            main_adddress[key] = ", ".join(value)

    if probability:
        main_adddress['good_proba'] = round(probability, 2)

    if address_dict.get(BUILDING_KEY):
        for building in address_dict.get(BUILDING_KEY):
            sub_address = main_adddress.copy()
            sub_address.update(building)
            sub_address[UNPARSED_KEY] = sub_address.pop(UNPARSED_KEY, [])
            result.append(sub_address)
    else:
        main_adddress[UNPARSED_KEY] = main_adddress.pop(UNPARSED_KEY, [])
        result.append(main_adddress)
    return result
