import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#==================================================================================
# 파일 가져오기 및 업로드
#==================================================================================
DIRECT_PATH=False

# 로컬에서 테스트 할때 편리
if DIRECT_PATH:
    # 파일 경로 직접 지정하기
    HOME = os.getcwd()
    trainPath = os.path.join(HOME, 'data/train.csv')
    df = pd.read_csv(trainPath)
    st.dataframe(df, use_container_width=False)

else:
    # 웹 서버에는 파일을 업로드 해줘야 함.
    # 파일 업로드 버튼 (업로드 기능)
    file = st.file_uploader("파일 선택(csv or excel)", type=['csv', 'xls', 'xlsx'])

    if file is not None:
        ext = file.name.split('.')[-1]
        if ext == 'csv':
            # 파일 읽기
            df = pd.read_csv(file)
            # 출력
        elif 'xls' in ext:
            # 엑셀 로드
            df = pd.read_excel(file, engine='openpyxl')
            # 출력
        
        st.dataframe(df)

#=============================================================================== 
# 파이썬 머신러닝 완벽 가이드 github 2장 코드 (참조) 그대로 사용
#===============================================================================
def get_category(age):
    cat = ''
    if age <= -1: cat = 'Unknown'
    elif age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else : cat = 'Elderly'
    
    return cat

def encode_features(dataDF):
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])  
    return dataDF

# Null 처리 함수
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)
    return df

# 머신러닝 알고리즘에 불필요한 피처 제거
def drop_features(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df

# 레이블 인코딩 수행.
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

# 앞에서 설정한 데이터 전처리 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

#==================================================================================
# 위의 코드를 그대로 가져와서 활용
#==================================================================================

def preProcess(titanic_df):
    y_titanic_df = titanic_df['Survived']
    X_titanic_df= titanic_df.drop('Survived',axis=1)

    X_titanic_df = transform_features(X_titanic_df)

    return X_titanic_df, y_titanic_df

plt.rcParams['font.family'] = "gulim"
plt.rcParams['axes.unicode_minus'] = False

#==================================================================================
# 시각화
#==================================================================================
# 성별에 따른 남성과 여성의 수 계산
gender_counts = df['Sex'].value_counts()

# Streamlit 대시보드 설정
st.title("타이타닉 생존자 예측")
st.header("탑승 승객의 남자와 여자의 수")

figsize=(4,4)

# Bar Chart
st.subheader("성별 분포 막대 그래프")
fig, ax = plt.subplots(figsize=figsize)
ax.bar(gender_counts.index, gender_counts.values)
ax.set_xlabel("성별")
ax.set_ylabel("인원수")
st.pyplot(fig)

# Pie Chart
st.subheader("성별 분포 원형 그래프")
fig, ax = plt.subplots(figsize=figsize)
ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)

#==================================================================================
# 데이터 전처리 및 데이터 분할
#==================================================================================
X_titanic_df, y_titanic_df = preProcess(df)
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, \
                                    test_size=0.2, random_state=11)


#==================================================================================
# streamlit으로 모델 선택
#==================================================================================
# 선택 박스
MODEL = st.selectbox(
    '어떤 모델을 사용하시겠습니까?',
    ('Decision-Tree', 'random-forest', 'logistic-regression'), 
    index=2
)
print(X_train.columns)
if MODEL == 'Decision-Tree':
    model = DecisionTreeClassifier(random_state=11)
    model.fit(X_train , y_train)
    dt_pred = model.predict(X_test)
    outStr = 'DecisionTreeClassifier 정확도: {0:.4f}'.format(accuracy_score(y_test, dt_pred))

elif MODEL == 'random-forest':
    model = RandomForestClassifier(random_state=11)
    model.fit(X_train , y_train)
    rf_pred = model.predict(X_test)
    outStr = 'RandomForestClassifier 정확도:{0:.4f}'.format(accuracy_score(y_test, rf_pred))

elif MODEL =='logistic-regression':
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train , y_train)
    lr_pred = model.predict(X_test)
    outStr = 'LogisticRegression 정확도: {0:.4f}'.format(accuracy_score(y_test, lr_pred))

st.write(outStr)
print(outStr)

#============================================================================
# 사용자 입력 받기
#============================================================================
pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=50.0)
cabin = st.text_input("Cabin", None)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# 입력 데이터 변환
sex = 1 if sex == "male" else 0
embarked_mapping = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_mapping[embarked]

# 예측 버튼
if st.button("Predict Survival"):
    # 입력 데이터로 예측하기
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Cabin': [cabin],
        'Embarked': [embarked]
    })
    print(input_data)
    # 모델로 생존 예측
    prediction = model.predict(input_data)[0]
    
    # 결과 출력
    if prediction == 1:
        st.success("The passenger is likely to survive.")
    else:
        st.error("The passenger is unlikely to survive.")