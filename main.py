import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from numpy.polynomial import Polynomial


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="기온 곡선 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 곡선 예측")

st.write(
    "서울 연평균 기온을 2005년 이전의 훈련 데이터로 학습하고, "
    "2005년 이후의 테스트 데이터로 1차·3차·9차 곡선의 예측 성능을 비교합니다."
)


# =========================================================
# 데이터 불러오기
# =========================================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/seoul.csv"
)

try:
    df = pd.read_csv(DATA_URL)

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.error(str(e))
    st.stop()


# =========================================================
# 필요한 열 확인
# =========================================================

if "날짜" not in df.columns:
    st.error("데이터에 '날짜' 열이 없습니다.")
    st.stop()

if "평균기온" not in df.columns:
    st.error("데이터에 '평균기온' 열이 없습니다.")
    st.stop()


# =========================================================
# 데이터 전처리
# =========================================================

df["날짜"] = pd.to_datetime(
    df["날짜"],
    errors="coerce"
)

df["평균기온"] = pd.to_numeric(
    df["평균기온"],
    errors="coerce"
)

df = df.dropna(
    subset=["날짜", "평균기온"]
).copy()

df["연도"] = df["날짜"].dt.year


# =========================================================
# 1906~2025년 사용
# =========================================================

df = df[
    (df["연도"] >= 1906) &
    (df["연도"] <= 2025)
].copy()


# =========================================================
# 연평균 기온 계산
# =========================================================

annual = (
    df.groupby("연도", as_index=False)["평균기온"]
    .mean()
    .rename(
        columns={
            "평균기온": "연평균기온"
        }
    )
)

annual = annual.sort_values(
    "연도"
).reset_index(drop=True)


if annual.empty:
    st.error("연평균 기온 데이터를 만들 수 없습니다.")
    st.stop()


# =========================================================
# 훈련 / 테스트 데이터 분리
#
# 2005년 이전 = 훈련
# 2005년부터 = 테스트
# =========================================================

train = annual[
    annual["연도"] < 2005
].copy()

test = annual[
    annual["연도"] >= 2005
].copy()


if train.empty:
    st.error("훈련용 데이터가 없습니다.")
    st.stop()

if test.empty:
    st.error("테스트용 데이터가 없습니다.")
    st.stop()


# =========================================================
# 연도 숫자 안정화
#
# 실제 연도:
# 1906, 1907, ...
#
# 계산용 연도:
# -99, -98, ...
#
# 2005년 = 0
# 2050년 = 45
#
# 고차 다항식 계산의 수치 안정성을 높이기 위해
# 연도를 2005년 기준으로 이동시킴
# =========================================================

TRAIN_X = (
    train["연도"].to_numpy(dtype=float)
    - 2005
)

TRAIN_Y = train[
    "연평균기온"
].to_numpy(dtype=float)

TEST_X = (
    test["연도"].to_numpy(dtype=float)
    - 2005
)

TEST_Y = test[
    "연평균기온"
].to_numpy(dtype=float)


# =========================================================
# 다항식 모델 만들기
#
# Polynomial.fit은 다항식 계산에서
# 수치적으로 안정적인 방식으로 적합
# =========================================================

def make_polynomial(
    x,
    y,
    degree
):

    model = Polynomial.fit(
        x,
        y,
        degree
    )

    return model


# =========================================================
# 평가 지표
# =========================================================

def calculate_mae(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    return np.mean(
        np.abs(
            actual - predicted
        )
    )


# =========================================================
# 1차 / 3차 / 9차 모델 학습
#
# 반드시 TRAIN_X, TRAIN_Y만 사용
# =========================================================

model_1 = make_polynomial(
    TRAIN_X,
    TRAIN_Y,
    1
)

model_3 = make_polynomial(
    TRAIN_X,
    TRAIN_Y,
    3
)

model_9 = make_polynomial(
    TRAIN_X,
    TRAIN_Y,
    9
)


# =========================================================
# 테스트 데이터 예측
#
# TEST_X는 학습에 사용하지 않았음
# =========================================================

pred_1_test = model_1(
    TEST_X
)

pred_3_test = model_3(
    TEST_X
)

pred_9_test = model_9(
    TEST_X
)


# =========================================================
# 테스트 데이터에서 MAE 계산
#
# 실제 테스트 데이터와 예측값의 차이
# =========================================================

mae_1 = calculate_mae(
    TEST_Y,
    pred_1_test
)

mae_3 = calculate_mae(
    TEST_Y,
    pred_3_test
)

mae_9 = calculate_mae(
    TEST_Y,
    pred_9_test
)


# =========================================================
# 2050년 예측
#
# 2050 - 2005 = 45
# =========================================================

year_2050_x = 2050 - 2005


prediction_2050_1 = float(
    model_1(year_2050_x)
)

prediction_2050_3 = float(
    model_3(year_2050_x)
)

prediction_2050_9 = float(
    model_9(year_2050_x)
)


# =========================================================
# 제목
# =========================================================

st.header("① 훈련용과 테스트용 데이터 분리")


st.write(
    "2005년 이전의 데이터만 모델을 학습하는 데 사용하고, "
    "2005년부터의 데이터는 학습에 사용하지 않은 채 "
    "마지막에 모델을 평가하는 데만 사용했습니다."
)


# =========================================================
# 데이터 개수 표시
# =========================================================

col1, col2 = st.columns(2)


with col1:

    st.metric(
        "훈련용 연도 수",
        f"{len(train)}개"
    )

    st.caption(
        f"{train['연도'].min()}~{train['연도'].max()}년"
    )


with col2:

    st.metric(
        "테스트용 연도 수",
        f"{len(test)}개"
    )

    st.caption(
        f"{test['연도'].min()}~{test['연도'].max()}년"
    )


# =========================================================
# 데이터 분리표
# =========================================================

split_table = pd.DataFrame({
    "구분": [
        "훈련용 데이터",
        "테스트용 데이터"
    ],
    "기간": [
        f"{train['연도'].min()}~{train['연도'].max()}",
        f"{test['연도'].min()}~{test['연도'].max()}"
    ],
    "연도 개수": [
        len(train),
        len(test)
    ],
    "모델 학습에 사용": [
        "O",
        "X"
    ]
})


st.dataframe(
    split_table,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 주의사항
# =========================================================

st.info(
    "중요: 테스트용 데이터(2005년 이후)는 1차·3차·9차 모델을 "
    "만들 때 사용하지 않았습니다. 모델을 모두 만든 뒤 "
    "테스트 데이터로 예측하고 MAE를 계산했습니다."
)


# =========================================================
# ② 곡선 비교 그래프
# =========================================================

st.header("② 1차·3차·9차 곡선 비교")


# 그래프용 연도
graph_years = np.linspace(
    1906,
    2050,
    500
)

# 계산용 연도
graph_x = (
    graph_years - 2005
)


graph_pred_1 = model_1(
    graph_x
)

graph_pred_3 = model_3(
    graph_x
)

graph_pred_9 = model_9(
    graph_x
)


fig = go.Figure()


# 실제 훈련 데이터
fig.add_trace(
    go.Scatter(
        x=train["연도"],
        y=train["연평균기온"],
        mode="markers",
        name="훈련용 실제값",
        marker=dict(size=5)
    )
)


# 실제 테스트 데이터
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="markers",
        name="테스트용 실제값",
        marker=dict(size=6)
    )
)


# 1차
fig.add_trace(
    go.Scatter(
        x=graph_years,
        y=graph_pred_1,
        mode="lines",
        name="1차 곡선(직선)",
        line=dict(width=3)
    )
)


# 3차
fig.add_trace(
    go.Scatter(
        x=graph_years,
        y=graph_pred_3,
        mode="lines",
        name="3차 곡선",
        line=dict(width=3)
    )
)


# 9차
fig.add_trace(
    go.Scatter(
        x=graph_years,
        y=graph_pred_9,
        mode="lines",
        name="9차 곡선",
        line=dict(width=3)
    )
)


# 2005년 기준선
fig.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="훈련 → 테스트",
    annotation_position="top"
)


# 2050년 기준선
fig.add_vline(
    x=2050,
    line_dash="dot",
    annotation_text="2050년",
    annotation_position="top"
)


fig.update_layout(
    title="훈련 데이터로만 학습한 1차·3차·9차 곡선",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# ③ 테스트 데이터 예측 성능
# =========================================================

st.header("③ 테스트 데이터 예측 성능")


st.write(
    "아래 MAE는 **학습에 사용하지 않은 2005년 이후 테스트 데이터**로 "
    "계산한 평균 절대 오차입니다."
)


# =========================================================
# 성능 카드
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.subheader("1차(직선)")

    st.metric(
        "테스트 MAE",
        f"{mae_1:.3f} ℃"
    )


with col2:

    st.subheader("3차 곡선")

    st.metric(
        "테스트 MAE",
        f"{mae_3:.3f} ℃"
    )


with col3:

    st.subheader("9차 곡선")

    st.metric(
        "테스트 MAE",
        f"{mae_9:.3f} ℃"
    )


# =========================================================
# ④ 최종 비교표
# =========================================================

st.header("④ 곡선별 테스트 성능과 2050년 예측")


result_table = pd.DataFrame({
    "모델": [
        "1차(직선)",
        "3차 곡선",
        "9차 곡선"
    ],
    "학습 기간": [
        f"{train['연도'].min()}~{train['연도'].max()}",
        f"{train['연도'].min()}~{train['연도'].max()}",
        f"{train['연도'].min()}~{train['연도'].max()}"
    ],
    "테스트 기간": [
        f"{test['연도'].min()}~{test['연도'].max()}",
        f"{test['연도'].min()}~{test['연도'].max()}",
        f"{test['연도'].min()}~{test['연도'].max()}"
    ],
    "테스트 MAE (℃)": [
        mae_1,
        mae_3,
        mae_9
    ],
    "2050년 예측 (℃)": [
        prediction_2050_1,
        prediction_2050_3,
        prediction_2050_9
    ]
})


st.dataframe(
    result_table.style.format({
        "테스트 MAE (℃)": "{:.3f}",
        "2050년 예측 (℃)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# ⑤ 테스트 실제값과 예측값
# =========================================================

st.header("⑤ 테스트 데이터에서 실제값과 예측값 비교")


test_result = test[
    ["연도", "연평균기온"]
].copy()


test_result["1차 예측"] = pred_1_test

test_result["3차 예측"] = pred_3_test

test_result["9차 예측"] = pred_9_test


fig_test = go.Figure()


# 실제값
fig_test.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["연평균기온"],
        mode="lines+markers",
        name="실제 기온",
        line=dict(width=3)
    )
)


# 1차
fig_test.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["1차 예측"],
        mode="lines",
        name="1차 예측",
        line=dict(
            width=2,
            dash="dash"
        )
    )
)


# 3차
fig_test.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["3차 예측"],
        mode="lines",
        name="3차 예측",
        line=dict(
            width=2,
            dash="dot"
        )
    )
)


# 9차
fig_test.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["9차 예측"],
        mode="lines",
        name="9차 예측",
        line=dict(
            width=2,
            dash="longdash"
        )
    )
)


fig_test.update_layout(
    title="학습에 사용하지 않은 테스트 데이터의 실제값과 예측값",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)


st.plotly_chart(
    fig_test,
    use_container_width=True
)


# =========================================================
# ⑥ 어떤 모델이 테스트에서 가장 잘했는가?
# =========================================================

st.header("⑥ 테스트 결과 해석")


mae_values = {
    "1차(직선)": mae_1,
    "3차 곡선": mae_3,
    "9차 곡선": mae_9
}


best_model = min(
    mae_values,
    key=mae_values.get
)


st.success(
    f"테스트 데이터의 MAE가 가장 작은 모델은 "
    f"**{best_model}**입니다. "
    f"평균적으로 약 **{mae_values[best_model]:.3f}℃** 정도 빗나갔습니다."
)


st.write(
    "MAE는 실제 기온과 예측 기온의 차이를 절댓값으로 계산한 "
    "평균이므로 **작을수록 테스트 예측 성능이 좋습니다.**"
)


# =========================================================
# ⑦ 2050년 예측 비교
# =========================================================

st.header("⑦ 2050년 기온 예측")


prediction_table = pd.DataFrame({
    "모델": [
        "1차(직선)",
        "3차 곡선",
        "9차 곡선"
    ],
    "2050년 예상 연평균기온 (℃)": [
        prediction_2050_1,
        prediction_2050_3,
        prediction_2050_9
    ]
})


st.dataframe(
    prediction_table.style.format({
        "2050년 예상 연평균기온 (℃)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# ⑧ 계산 방법
# =========================================================

st.header("⑧ 이번 분석에서 중요한 점")


st.markdown(
    """
### 훈련 데이터
**2004년까지**의 연평균 기온을 사용했습니다.

### 테스트 데이터
**2005년부터**의 연평균 기온을 사용했습니다.

테스트 데이터는 모델을 만드는 과정에서는 전혀 사용하지 않고,
마지막에 모델의 예측 성능을 평가할 때만 사용했습니다.

### 1차
직선 형태의 모델입니다.

### 3차
한 번 휘어지는 정도보다 더 복잡한 곡선으로
기온의 변화 패턴을 표현할 수 있습니다.

### 9차
훨씬 복잡한 곡선을 만들 수 있습니다.
하지만 복잡한 모델이 항상 미래 예측을 잘하는 것은 아닙니다.

### 2050년 예측
세 모델을 **2004년까지의 데이터로만 학습한 뒤**
2050년을 넣어서 계산했습니다.

따라서 2050년 값은 실제 관측값이 아니라
각 곡선이 추정한 **예측값**입니다.
"""
)


# =========================================================
# 데이터 처리 기준
# =========================================================

with st.expander("📌 데이터 처리 기준"):

    st.write(
        """
        - 서울 기온 데이터(seoul.csv)를 사용했습니다.
        - 날짜를 날짜 형식으로 변환했습니다.
        - 평균기온을 숫자로 변환했습니다.
        - 1906~2025년 데이터를 사용했습니다.
        - 일별 평균기온을 연도별로 평균하여 연평균 기온을 계산했습니다.
        - 2005년 이전(2004년까지)을 훈련 데이터로 사용했습니다.
        - 2005년부터를 테스트 데이터로 사용했습니다.
        - 1차, 3차, 9차 다항식을 훈련 데이터에만 맞췄습니다.
        - 테스트 데이터는 학습 과정에 사용하지 않았습니다.
        - 테스트 데이터에서 MAE를 계산했습니다.
        - 2050년은 각 모델에 입력하여 예측했습니다.
        - 고차 다항식의 수치적 불안정을 줄이기 위해
          계산용 연도를 '연도 - 2005'로 변환했습니다.
        - scikit-learn은 사용하지 않고 numpy를 사용했습니다.
        """
    )
