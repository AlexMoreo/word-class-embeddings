import numpy as np
from tqdm import tqdm
import re

def text_processing(text):
    return text.lower()

def process_MDS(data, text_processor=text_processing):
    for domain in tqdm(data.keys()):
        for collection in data[domain]:
            data[domain][collection] = [(text_processor(text),label) for text,label in data[domain][collection]]

def stats(data):
    lenghts = []
    for domain in tqdm(data.keys()):
        for collection in data[domain]:
            lenghts.extend([len(text.split()) for text,_ in data[domain][collection]])
    print(f'mean={np.mean(lenghts):.3f}')
    print(f'std={np.std(lenghts):.3f}')
    print(f'max={np.max(lenghts):.0f}')
    print(f'min={np.min(lenghts):.0f}')

def mask_numbers(data, number_mask='numbermask'):
    mask = re.compile(r'\b[0-9][0-9.,-]*\b')
    masked = []
    for text in tqdm(data, desc='masking numbers'):
        masked.append(mask.sub(number_mask, text))
    return masked

