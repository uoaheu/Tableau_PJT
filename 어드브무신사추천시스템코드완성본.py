#%%
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, accuracy

import warnings; warnings.simplefilter('ignore')

from konlpy.tag import Mecab 
from konlpy.tag import Komoran
import re
# %%
def preprocess(text):
    text = str(text)
    text = ' '.join(re.findall(r'[가-힣]+', text))                     # 한국어 단어만 가져옴 (이 작업 수행하지 않으면 unicode 에러 발생 가능)
    komoran = Komoran()
    tokens = komoran.nouns(text)                                      # 명사만 가져옴
    tokens = [t for t in tokens if re.fullmatch(r'[가-힣]{2,}', t)]    # 2글자 이상 단어 선택
    return ' '.join(tokens)
# %%
## 코사인 유사도 - 추천

# %%
df = pd.read_csv('/Users/hongnayeon/Desktop/ADV_무신사/df_img.csv', encoding='utf-8')
# %%
documents_preprocessed = list(df['documents_preprocessed'].astype('str'))
documents_preprocessed
# %%
#문서단어행렬 만들기

stop_words = ['가가']

tfidfVectorizer = TfidfVectorizer(min_df=2, max_df=0.5, stop_words=stop_words)   # 2개 이상 전체 문서의 50% 이하의 문서에서 출현한 단어만 고려
feat_vect = tfidfVectorizer.fit_transform(documents_preprocessed)
feature_names = tfidfVectorizer.get_feature_names_out()
print('단어 목록:', feature_names)
print('문서단어행렬 모양 (문서의 수, 단어의 수):', feat_vect.shape)
print(feat_vect.toarray())
# %%
def hybrid(review):
    import json
    # 리뷰 전처리
    review_preprocessed = preprocess(review)
    
    # 전처리 된 리뷰 TF-IDF vector로 바꾸기
    review_vector = tfidfVectorizer.transform([review_preprocessed])

    # 리뷰 벡터와 상품 리뷰 벡터 간 코사인 유사도 계산
    sim_scores = cosine_similarity(feat_vect, review_vector) 
    
    # 모든 상품에 대한 유사도 점수 가져오기
    product_sim_scores = list(enumerate(sim_scores))
    product_sim_scores = sorted(product_sim_scores, key=lambda x: x[1], reverse=True)
    product_sim_scores = product_sim_scores[1:1000]    # 100까지 했다가 신발이 안나오는 경우가 있어, 그냥 대폭 늘렸음

    # 가장 유사한 상품의 인덱스
    product_indices = [i[0] for i in product_sim_scores]
    
    # 상품 정보들 가져오기
    products = df.iloc[product_indices][['MainCategory','ItemName', 'MainImage']]  # 확인할려고 UserText 같이 넣은거
    
    # 리뷰_벡터를 사용하여 예상 / 평점 기준으로 영화 정렬
    products['est'] = products.index.map(lambda x: cosine_similarity(feat_vect[x], review_vector)[0][0])
    products = products.sort_values('est', ascending=False)
  
    top_p = products.groupby('MainCategory').head(1)

#     top_dic = top_p.to_dict(orient='records')
#     json_val = json.dumps(top_dic)
    top_p_json = top_p.to_json(orient='records', force_ascii=False)
    
    return top_p
# %%
review_text = "따뜻하고 편해요. 집 주변에 편하게 입고 나갈 옷을 원해요"
r = hybrid(review_text)
r
# %%
def OuterName(review):
    df = hybrid(review)
    outerName = df.loc[df['MainCategory']=='아우터']
    r = outerName['ItemName'].to_string()
    r = r.split(maxsplit = 1)
    r = r[1]
    return r
# %%
def TopName(review):
    df = hybrid(review)
    TopName = df.loc[df['MainCategory']=='상의']
    r = TopName['ItemName'].to_string()
    r = r.split(maxsplit = 1)
    r = r[1]
    return r
# %%
def BottomName(review):
    df = hybrid(review)
    BottomName = df.loc[df['MainCategory']=='바지']
    r = BottomName['ItemName'].to_string()
    r = r.split(maxsplit = 1)
    r = r[1]
    return r
# %%
review_text = "따뜻하고 편해요. 집 주변에 편하게 입고 나갈 옷을 원해요"
r = OuterName(review_text)
r
# %%
review_text = "따뜻하고 편해요. 집 주변에 편하게 입고 나갈 옷을 원해요"
r = BottomName(review_text)
r
# %%
def OuterImg(review):
    import pandas
    pd.options.display.max_colwidth = 2000 # 이미지 url 잘리지 않고 전체 줄력되게끔 설정
    
    df = hybrid(review)
    outerImg = df.loc[df['MainCategory']=='아우터']
    r = outerImg['MainImage'].to_string()
    r = r.split()
    r = r[1]
    
    return r
# %%
def TopImg(review):
    import pandas
    pd.options.display.max_colwidth = 2000 # 이미지 url 잘리지 않고 전체 줄력되게끔 설정
    
    df = hybrid(review)
    TopImg = df.loc[df['MainCategory']=='상의']
    r = TopImg['MainImage'].to_string()
    r = r.split()
    r = r[1]
    
    return r
# %%
def BottomImg(review):
    import pandas
    pd.options.display.max_colwidth = 2000 # 이미지 url 잘리지 않고 전체 줄력되게끔 설정
    
    df = hybrid(review)
    BottomImg = df.loc[df['MainCategory']=='바지']
    r = BottomImg['MainImage'].to_string()
    r = r.split()
    r = r[1]
    
    return r
# %%
review_text = "따뜻하고 편해요. 집 주변에 편하게 입고 나갈 옷을 원해요"
r = OuterImg(review_text)
r
# %%
from tabpy.tabpy_tools.client import Client

client = Client('http://localhost:9004/')
# %%
client.deploy('hybrid', hybrid, '추천 모델', override = True)
client.deploy('OuterName', OuterName, '추천할 아우터 이름', override = True)
client.deploy('TopName', TopName, '추천할 상의 이름', override = True)
client.deploy('BottomName', BottomName, '추천할 바지 이름', override = True)
client.deploy('OuterImg', OuterImg, '추천할 아우터 이미지', override = True)
client.deploy('TopImg', TopImg, '추천할 상의 이미지', override = True)
client.deploy('BottomImg', BottomImg, '추천할 바지 이미지', override = True)
# %%
