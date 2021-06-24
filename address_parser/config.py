from pathlib import Path

project_path = Path(__file__).resolve().parent
models_path = project_path / 'models'

model_number = 12
CHAR_PATH = models_path / f'char_vocab_{model_number}.json'
LABEL_PATH = models_path / f'label2id_{model_number}.json'
MODEL_SIZE_PATH = models_path / f'model_size_{model_number}.json'
MODEL_PATH = models_path / f'sentence_level_pos_{model_number}.pth'

TFIDF_VECTORIZER_PATH = models_path / 'tfidf_vectorizer_1.pkl'
GOOD_CLASSIFIER_PATH = models_path / 'classifier_1.pkl'

BUILDING_ENTITY = {
    'house', 'house_type',
    'corpus', 'corpus_type',
    'block', 'block_type',
    'block-section', 'block-section_type',
    'liter', 'liter_type',
    'section', 'section_type',
    'structure', 'structure_type',
}

BUILDING_KEY = 'building'
COMMA_TAG = 'comma'
UNPARSED_KEY = 'unparsed_parts'
STREET_KEY = 'street'
STREET_TYPE_KEY = 'street_type'
SEVERAL_STREETS = 'сочетание'

LEMMA = {
    'region_type': {
        'обл': 'область',
        'кр': 'край',
        'респ': 'республика',
        'ао': 'автономный округ',
        'г': 'город',
        'гор': 'город',
        'аобл': 'автономная область',
    },
    'area_type': {
        'р-н': 'район',
        'района': 'район',
    },
    'city_type': {
        'г': 'город',
        'гор': 'город',
    },
    'settlement_type': {
        'д': 'деревня',
        'c': 'село',
        'п': 'поселок',
        'пгт': 'поселок городского типа',
        'рп': 'рабочий поселок',
    },
    'district_type': {
        'р-н': 'район',
        'района': 'район',
        'ж/р': 'жилой район',
        'жилого района': 'жилой район',
    },
    'microdistrict_type': {
        'мкр': 'микрорайон',
        'мкрн': 'микрорайон',
        'мкр-не': 'микрорайон',
        'мкр-н': 'микрорайон',
        'жил микрорайон': 'жилой микрорайон',
        'м-рн': 'микрорайон',
        'микрорайона': 'микрорайон',
        'микрорайоне': 'микрорайон',
    },
    'quarter_type': {
        'жил квартал': 'жилой квартал',
        'кврт': 'квартал',
        'кв-л': 'квартал',
        'жк': 'жилой квартал',
        'жил комплекс': 'жилой комплекс',
        'жил массив': 'жилой массив',
        'окр': 'округ',
    },
    'street_type': {
        'ул': 'улица',
        'пр-д': "проезд",
        'пр-кт': "проспект",
        'пр': "проспект",
        'пер': 'переулок',
        'пл': 'площадь',
        'ш': 'шоссе',
        'шос': 'шоссе',
        'мгстр': 'магистраль',
        'м': 'магистраль',
        'ал': 'аллея',
        'тер': 'территория',
    },
    'house_type': {
        'жилой дом': 'дом',
        'жил дом': 'дом',
        'д': 'дом',
        'жилого дома': 'дом',
        'поз': 'позиция',
        'влад': 'владение',
        'влд': 'владение',
        'вл': 'владение',
    },
    'corpus_type': {
        'кор': 'корпус',
        'корп': 'корпус',
        'крп': 'корпус',
        'к': 'корпус',
    },
    'structure_type': {
        'стр': 'строение',
        'с': 'строение',
    },
    'section_type': {
        'сек': 'секция',
        'с': 'секция',
    },
    'queue_type': {
        'оч': 'очередь',
    },
    'liter_type': {
        'литер': 'литера',
        'лит': 'литера',
    },
    'block_type': {
        'блк': 'блок',
    },
    'block-section_type': {

    },
    'stage_type': {

    },
    'sector_type': {
        'уч': 'участок',
    },
    'gp_type': {
        'гп': 'генеральный план',
        'генплан': 'генеральный план',
        'ген план': 'генеральный план',
    },
}

