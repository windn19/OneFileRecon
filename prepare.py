from collections import Counter
from datetime import datetime
import json
import logging
import logging.config
import re
import os


number_lat = re.compile(r'[ABEKMHOPCTYX_]\d{3}[ABEKMHOPCTYX_]{2}\d{2,3}|[ABEKMHOPCTYX_]{2}\d{5,6}')
# num_lat = re.compile(r'[ABEKMHOPCTYX_]\d{3}[ABEKMHOPCTYX_]{2}|[ABEKMHOPCTYX_]{2}\d{5,6}')
number_by = re.compile(r'\d{4}[A-Z]{2}-\d')
number_by1 = re.compile(r'\d?\w{3}\d{3,4}')
number_by2 = re.compile(r'\d{2}\w{2}\d{3}')
number_lat1 = re.compile(r'\w{2}\d{5}')
number_lat2 = re.compile(r'\w{2}\d{4}\w')
number_tj = re.compile(r'\d{5}\w{3}')
# logging.config.fileConfig('file.conf')
log = logging.getLogger('file1')


def prepare1(source):
    log.info(source)
    source = source.replace('[UNK]', '_')
    res = number_lat.findall(source)
    res1 = number_by.findall(source)
    res2 = number_by1.findall(source)
    res3 = number_by2.findall(source)
    res4 = number_lat1.findall(source)
    res5 = number_lat2.findall(source)
    res6 = number_tj.findall(source)
    if not res:
        log.info('По иностранным шаблонам')
        out = Counter(res1 + res2 + res3 + res4 + res5 + res6)
    elif len(res):
        log.info('Российский номер')
        out = res[0]
    else:
        return 'Неопознан'
    log.info(f'Результат: {source} - {out}')
    if not out:
        return 'Неопознан'
    if isinstance(out, str):
        return out
    else:
        return max(out.keys(), key=lambda x: out[x])
     
       
