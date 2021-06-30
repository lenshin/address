# Russian address parser
The service parses Russian post addresses into named entities. 
The service supports any order of address writing.
For more examples please see 1.0-example.ipynb

The parser support following address entities:
- zipcode - number of zip code
- zipcode_type - type of zip code
- region - name of the region
- region_type - type of the region
- area - name of area in the region
- area_type - type of area in the region
- city - city name
- city_type - city type
- settlement - name of the settlement
- settlement_type - type of the settlement
- district - name of area in city or settlement
- district_type - type of area in city or settlement
- microdistrict - name of the microdistrict
- microdistrict_type - name of the microdistrict
- quarter - name of the quarter
- quarter_type - type of the quarter
- street - name of the street
- street_type - type of the street
- house - number of the house
- house_type - type of the house
- corpus - number of the corpus
- corpus_type - type of the corpus
- rm - number of apartment or office
- rm_type - type of apartment or office

## Start-up instructions
1) Clone the repository and switch to directory

```
git clone https://github.com/lenshin/address.git
cd address
```

2) Install requirements

```
pip install -r requirements.txt
```

3) Running-up service

```python
import sys
sys.path.append("PUT HERE PATH TO REPOSITORY")

from address_parser import AddressParser

parser = AddressParser()

address = '123456, Тверская область, Тверь, ул. Оснабрюкская, д. 10 к.1, кв. 45'
parser(address)
```
```
[{'zipcode': '123456',
  'region': 'тверская',
  'region_type': 'область',
  'city': 'тверь',
  'street_type': 'улица',
  'street': 'оснабрюкская',
  'rm_type': 'оф',
  'rm': '45',
  'house_type': 'дом',
  'house': '10',
  'corpus_type': 'корпус',
  'corpus': '1',
  'unparsed_parts': []}]
```

