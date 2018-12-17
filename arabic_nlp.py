import os
from nltk.parse import stanford as SParse
from nltk.tag import stanford as STag
from nltk.tokenize import StanfordSegmenter

from polyglot.text import Text
from rake_nltk import Rake

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

os.environ['STANFORD_MODELS'] = 'stanford-segmenter-2018-10-16/data/;stanford-postagger-full-2018-10-16/models/'
os.environ['STANFORD_PARSER'] = 'stanford-parser-full-2018-10-17'
os.environ['CLASSPATH'] = 'stanford-parser-full-2018-10-17'
os.environ['JAVAHOME'] = 'C:/Program Files/Java/jdk-11.0.1'

segmenter = StanfordSegmenter('stanford-segmenter-2018-10-16/stanford-segmenter-3.9.2.jar')
segmenter.default_config('ar')
text = segmenter.segment_file('sample.txt')
print(text)

tagger = STag.StanfordPOSTagger('arabic.tagger', 'stanford-postagger-full-2018-10-16/stanford-postagger.jar')
for tag in tagger.tag(text.split()):
    print(tag[1])
    
parser = SParse.StanfordParser(model_path='edu/stanford/nlp/models/lexparser/arabicFactored.ser.gz')
sentences = parser.raw_parse_sents(text.split('.'))
for line in sentences:
    for sentence in line:
        print(sentence)
        sentence.draw()
        
ner = Text(text)
for sent in ner.sentences:
    print(sent)
    for entity in sent.entities:
        print(entity.tag, entity)
    print('')
    
with open('ar_london.txt', encoding='utf-8') as f:
    london = f.read()
print(london[:100])

rake = Rake(stopwords=stopwords.words('arabic'), punctuations=',./:،؛":.,’\''.split(), language='arabic', max_length=15)
rake.extract_keywords_from_text(london)
for phrase in rake.get_ranked_phrases()[:5]:
    print(phrase)