import numpy as np
import pandas as pd
import streamlit as st
import random
import string
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings

warnings.filterwarnings('ignore')

# Data read
pd.set_option('display.max_columns', 500)
bath = 'C:/Users/dmsco/Documents/GitHub/C.lab_recommendation/data/'
cus_df = pd.read_excel(bath + 'Customer_Data_ori.xlsx')
map_df = pd.read_csv(bath + 'course_data_0124.csv')

# 시기별 데이터를 병합
new_data = {
    '일련번호': list(range(1, 33)),
    '시기별': ['0,72', '0,24', '22,27', '34,39', '45,48', '13,24', '25,36', '37,48', '0,36',
               '37,72', '0,72', '0,36', '37,72', '0,72', '0,1', '0,12', '13,72', '12,35', '24,47',
               '48,72', '0,12', '24,47', '47,72', '0,24', '25,35', '36,47', '25,36', '36,72', 
               '24,72', '24,72', '24,72', '0,72']
}

temp_df = pd.DataFrame(new_data)
map_df = pd.merge(map_df, temp_df, on='일련번호')


def generate_random_id(length=8, random_state=42):
    random.seed(random_state)
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for _ in range(length))


def preprocess_customer_data(cus_df):
    # 고객 id 컬럼 생성
    cus_df['고객id'] = [generate_random_id() for _ in range(len(cus_df))]

    # 고객id 컬럼을 Name 컬럼 앞에 위치시키기
    cols = cus_df.columns.tolist()
    cols.insert(cols.index('Name'), cols.pop(cols.index('고객id')))
    cus_df = cus_df[cols]

    # cus_df의 Parenting_Concerns 컬럼 전처리
    cus_df['Parenting_Concerns'] = cus_df['Parenting_Concerns'].str.strip("[]").str.replace("'", "").str.replace(".", ",")
    
    return cus_df


def preprocess_map_data(map_df):
    # 원하는 순서대로 컬럼명을 나열
    new_column_order = [
        '일련번호', '주제    구분', '내용구분', '시기별', '시기', '시기_2', '코스 명', '3T', 'NVC', '육아정보',
        '양육자Only', '추천태그', '관련 육아고민', '세부내용', '기간(week)', 'Score_3T',
        'Score_NVC', 'Score_육아정보', 'Total_Score'
    ]

    # 컬럼 순서를 변경하여 데이터프레임 재정렬
    map_df = map_df[new_column_order]

    # 불필요한 컬럼 드랍
    map_df.drop([
        '주제    구분', '3T', 'NVC', '육아정보', '시기', '시기_2', '양육자Only', '기간(week)',
        'Score_3T', 'Score_NVC', 'Score_육아정보', 'Total_Score'
    ], axis=1, inplace=True)
    
    return map_df


# 전처리 함수 정의
def preprocess_text(text):
    text = re.sub(r'\[/()+.*?\]', '', text)
    text = re.sub(r'\W+', ' ', text)
    stopwords = ['의', '세', '저', '제', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    text = ' '.join(word for word in text.split() if word not in stopwords)
    return text


def recommend_courses(course_names, child_age_months, map_df):
    # 전처리 단계: Null 값 대체
    selected_features = ['추천태그', '관련 육아고민']
    for feature in selected_features:
        map_df[feature] = map_df[feature].fillna('')

    # 'combined_features' 컬럼 생성 및 전처리 적용
    map_df['combined_features'] = map_df['추천태그'] + ' ' + map_df['관련 육아고민']
    map_df['combined_features'] = map_df['combined_features'].apply(preprocess_text)

    # TF-IDF 벡터화 및 코사인 유사도 계산
    tfidf_vectorizer = TfidfVectorizer()
    feature_vectors = tfidf_vectorizer.fit_transform(map_df['combined_features'])

    # 입력값 전처리 및 벡터화
    processed_course_name = preprocess_text(course_names)
    input_vector = tfidf_vectorizer.transform([processed_course_name])

    # 코사인 유사도 계산
    similarity = cosine_similarity(input_vector, feature_vectors)

    # 유사도 점수와 인덱스를 결합
    similarity_scores = list(enumerate(similarity[0]))

    # 유사도 점수 정렬
    sorted_similar_courses = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # 유사한 상위 5개 항목 선택
    top_sim = sorted_similar_courses[:5]

    # st.write("Top similar courses based on input:")
    # for i, concern in enumerate(top_sim):
    #     index = concern[0]
    #     concern_from_index = map_df.loc[index, '코스 명']
    #     st.write(f"{i + 1} - {concern_from_index}")

    # 아이의 개월 수로 코스 필터링
    def filter_courses_by_age(child_age_months, map_df):
        def is_within_age_range(age_range, child_age):
            # Convert the age_range string to a list of two integers
            min_age, max_age = map(int, age_range.split(","))
            return min_age <= child_age <= max_age

        # Filter the map_df based on whether the child age is within the age range in 시기별
        filtered_df = map_df[map_df['시기별'].apply(lambda x: is_within_age_range(x, child_age_months))]
        return filtered_df

    # 아이의 개월 수로 코스 필터링
    filtered_courses = filter_courses_by_age(child_age_months, map_df)

    # 필요한 컬럼 선택 및 인덱스 제거
    result_columns = ['코스 명', '기간(week)', '추천태그', '세부내용', '양육자Only' ]
    filtered_courses = filtered_courses[result_columns].reset_index(drop=True)

    return filtered_courses.head()


# 데이터 전처리
cus_df = preprocess_customer_data(cus_df)
map_df = preprocess_map_data(map_df)


def main():
    st.title("👼육아 코스 추천👼")

    # 사용자 입력 받기
    # 육아 고민 선택 옵션
    concerns_options = ['필수 육아관',  '태담','생애초기',
        '대화법',  '생활습관', '갈등중재', '상호작용', '발달사항', '언어자극','육아대화',  '육아관', 
        '칭찬',  '싫어', '애착', '식습관', '그만봐',  '사교육', '어린이집', '자기공감', 
        '거절',  '샤워', '미디어', '집안일',  '등하원',  '감정표현', 
        '배변', '디지털', '놀이터', '육아방법',  '훈육상황', '놀이', '공격성', '분리불안', 
        '건강', '수면', '공감 육아방법', '발달',  '관계',  '감사',  '떼쓰기', '훈육', '약속',
        '산후우울','육아번아웃','초보부모', '육아스트레스', '우울감','가사분담 육아관','부모역할','부부대화', '육아분담',
        '연년생','형제','남매', '미운네살', '아들딸', '쌍둥이'
    ]

    selected_concerns = st.multiselect('당신의 육아 고민은 무엇입니까?', concerns_options, help="육아 고민을 최소 3개 이상 선택해주세요.")
    course_names = ' '.join(selected_concerns)
    
    child_age_months = st.number_input('아이의 나이(개월 수)를 입력해주세요. : ', min_value=0, max_value=72, step=1, help="아이의 나이는 0부터 72개월 사이로 입력해주세요.")

    if st.button('추천 코스 찾기'):
        if len(selected_concerns) >= 3 and child_age_months is not None:
            # 추천 코스 출력
            recommended_courses = recommend_courses(course_names, int(child_age_months), map_df)
            st.write("다음과 같은 코스가 추천되었습니다.")
            st.write(recommended_courses)
        else:
            st.write("육아 고민을 최소 3개 이상 선택하고 아이의 나이를 입력해주세요.")
            


if __name__ == '__main__':
    main()
