import pandas as pd
import re
def clean(s):
    s = re.sub(r"([.!?,'/()])", r" \1 ", s)
    s = re.sub(r"[اأإآءئ]", "ا", s)
    s = re.sub(r"[هة]", "ه", s)
    return s
df = pd.read_csv('unbalanced-reviews.txt', sep='\t', encoding='utf-16')
df.review = df.review.apply(clean)
df['label'] = '__label__' + df.rating.astype(str)


from sklearn.model_selection import StratifiedShuffleSplit
spl = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=123)
for tr_ix, te_ix in spl.split(df.review, df.label):
    break
with open('hotel.train', 'w', encoding='utf-8') as f:
    for lbl, txt in df[['label', 'review']].iloc[tr_ix].values:
        print(lbl, txt, file=f)
with open('hotel.test', 'w', encoding='utf-8') as f:
    for lbl, txt in df[['label', 'review']].iloc[te_ix].values:
        print(lbl, txt, file=f)
        
        
# run the following in command line:
'''
fasttext.exe supervised -input hotel.train -output model_hotel -wordNgrams 2
fasttext.exe test model_hotel.bin hotel.test
'''