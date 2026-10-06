import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")

st.write(
    "서울의 연평균 기온 데이터를 이용해 "
    "기온 변화 추세를 선형회귀로 분석하고 미래 기온을 예측합니다."
)


# =========================================================
# 데이터 불러오기
# =========================================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/seoul.csv"
)

try:
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.error(str(e))
    st.stop()


# =========================================================
# 필요한 열 확인
# =========================================================

required_columns = [
    "날짜",
    "평균기온"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        f"다음 열이 데이터에 없습니다: {missing_columns}"
    )
    st.write("현재 데이터의 열:")
    st.write(df.columns.tolist())
    st.stop()


# =========================================================
# 원본 데이터 전처리
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
# 2025년 이후 데이터 제외
# =========================================================

df = df[
    df["연도"] <= 2025
].copy()


# =========================================================
# 연도별 관측일수 계산
# 평균기온이 300일 이상 있는 연도만 사용
# =========================================================

observation_count = (
    df.groupby("연도")["평균기온"]
    .count()
    .reset_index(name="관측일수")
)

valid_years = observation_count[
    observation_count["관측일수"] >= 300
]["연도"]


df = df[
    df["연도"].isin(valid_years)
].copy()


# =========================================================
# 연평균 기온 계산
# =========================================================

annual = (
    df.groupby("연도")["평균기온"]
    .mean()
    .reset_index()
)

# 원본의 평균기온과 구분하기 위해 이름 변경
annual = annual.rename(
    columns={
        "평균기온": "연평균기온"
    }
)

annual = annual.sort_values(
    "연도"
).reset_index(drop=True)


# =========================================================
# 전체 연도 범위에 맞춰 데이터 준비
# =========================================================

if len(annual) == 0:
    st.error("연평균 기온 데이터를 만들 수 없습니다.")
    st.stop()


# =========================================================
# 회귀 모델 함수
# =========================================================

def make_regression(data):

    X = data[["연도"]]
    y = data["연평균기온"]

    model = LinearRegression()
    model.fit(X, y)

    return model


# =========================================================
# 전체 기간 회귀
# =========================================================

full_model = make_regression(
    annual
)

full_slope = full_model.coef_[0]

full_intercept = full_model.intercept_

full_slope_100 = full_slope * 100


annual["전체회귀예측"] = full_model.predict(
    annual[["연도"]]
)


# =========================================================
# 최근 20년 회귀
# =========================================================

recent_20 = annual[
    (annual["연도"] >= 2006) &
    (annual["연도"] <= 2025)
].copy()


recent_model = make_regression(
    recent_20
)

recent_slope = recent_model.coef_[0]

recent_intercept = recent_model.intercept_

recent_slope_100 = recent_slope * 100


# =========================================================
# 전체 기간 회귀식
# =========================================================

st.header("1. 전체 기간의 기온 변화")

st.subheader("전체 기간 회귀선")

st.latex(
    f"y = {full_slope:.4f}x + {full_intercept:.2f}"
)

st.metric(
    "전체 기간 기울기",
    f"100년에 {full_slope_100:+.2f} ℃"
)


# =========================================================
# 최근 20년 회귀식
# =========================================================

st.subheader("최근 20년 회귀선")

st.latex(
    f"y = {recent_slope:.4f}x + {recent_intercept:.2f}"
)

st.metric(
    "최근 20년 기울기",
    f"100년에 {recent_slope_100:+.2f} ℃"
)


# =========================================================
# 전체 기간 vs 최근 20년 기울기 비교
# =========================================================

st.subheader("전체 기간과 최근 20년의 기울기 비교")

slope_compare = pd.DataFrame({
    "기간": [
        "전체 기간",
        "최근 20년"
    ],
    "100년당 기온 변화 (℃)": [
        full_slope_100,
        recent_slope_100
    ]
})

st.dataframe(
    slope_compare.style.format({
        "100년당 기온 변화 (℃)": "{:+.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 전체 기간 그래프
# =========================================================

fig_main = go.Figure()


# 실제 연평균 기온
fig_main.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균 기온",
        marker=dict(size=6)
    )
)


# 전체 기간 회귀선
fig_main.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["전체회귀예측"],
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(width=3)
    )
)


# 최근 20년 회귀선
recent_x = np.arange(
    2006,
    2026
)

recent_y = recent_model.predict(
    pd.DataFrame({
        "연도": recent_x
    })
)

fig_main.add_trace(
    go.Scatter(
        x=recent_x,
        y=recent_y,
        mode="lines",
        name="최근 20년 회귀선",
        line=dict(
            width=3,
            dash="dash"
        )
    )
)


fig_main.update_layout(
    title="서울 연평균 기온과 회귀선",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)

st.plotly_chart(
    fig_main,
    use_container_width=True
)


# =========================================================
# 회귀선 해석
# =========================================================

st.info(
    f"""
전체 기간에서는 100년에 약 **{full_slope_100:+.2f}℃**의 변화 추세가 나타납니다.

최근 20년에서는 100년에 약 **{recent_slope_100:+.2f}℃**의 변화 추세가 나타납니다.

따라서 두 기간의 기울기를 비교하면 최근 기온 변화 추세가
장기적인 추세와 어떻게 다른지 확인할 수 있습니다.
"""
)


# =========================================================
# 미래 연도 예측
# =========================================================

st.header("2. 연도를 선택하여 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2050,
    step=1
)


future_prediction = full_model.predict(
    pd.DataFrame({
        "연도": [selected_year]
    })
)[0]


st.metric(
    f"{selected_year}년 예상 연평균 기온",
    f"{future_prediction:.2f} ℃"
)

st.caption(
    "이 값은 전체 기간의 선형회귀 추세를 단순히 연장한 예측값입니다."
)


# =========================================================
# 미래 예측 그래프
# =========================================================

prediction_years = np.arange(
    1900,
    2101
)

prediction_values = full_model.predict(
    pd.DataFrame({
        "연도": prediction_years
    })
)


fig_prediction = go.Figure()


fig_prediction.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(size=5)
    )
)


fig_prediction.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_values,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(width=3)
    )
)


fig_prediction.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[future_prediction],
        mode="markers",
        name=f"{selected_year}년 예측",
        marker=dict(size=12)
    )
)


fig_prediction.update_layout(
    title="1900~2100년 기온 변화 추세와 예측",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)


st.plotly_chart(
    fig_prediction,
    use_container_width=True
)


# =========================================================
# =========================================================
# 새로 추가한 선형회귀 모델 평가
# =========================================================
# =========================================================

st.header(
    "3. 50년 학습과 100년 학습의 예측 성능 비교"
)

st.write(
    "과거 데이터를 학습 데이터로 사용하고, "
    "학습에 사용하지 않은 최근 20년(2006~2025)을 "
    "공통 테스트 데이터로 사용합니다."
)


# =========================================================
# 50년 학습 데이터
# =========================================================

train_50 = annual[
    (annual["연도"] >= 1956) &
    (annual["연도"] <= 2005)
].copy()


# =========================================================
# 100년 학습 데이터
# =========================================================

train_100 = annual[
    (annual["연도"] >= 1906) &
    (annual["연도"] <= 2005)
].copy()


# =========================================================
# 공통 테스트 데이터
# =========================================================

test_20 = annual[
    (annual["연도"] >= 2006) &
    (annual["연도"] <= 2025)
].copy()


# =========================================================
# 데이터 개수 확인
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "50년 학습 데이터",
        f"{len(train_50)}년"
    )

with col2:
    st.metric(
        "100년 학습 데이터",
        f"{len(train_100)}년"
    )

with col3:
    st.metric(
        "공통 테스트 데이터",
        f"{len(test_20)}년"
    )


# =========================================================
# 50년 모델
# =========================================================

model_50 = make_regression(
    train_50
)

pred_50 = model_50.predict(
    test_20[["연도"]]
)


# =========================================================
# 100년 모델
# =========================================================

model_100 = make_regression(
    train_100
)

pred_100 = model_100.predict(
    test_20[["연도"]]
)


# =========================================================
# 50년 모델 성능 평가
# =========================================================

y_test = test_20["연평균기온"]

mae_50 = mean_absolute_error(
    y_test,
    pred_50
)

mse_50 = mean_squared_error(
    y_test,
    pred_50
)

r2_50 = r2_score(
    y_test,
    pred_50
)


# =========================================================
# 100년 모델 성능 평가
# =========================================================

mae_100 = mean_absolute_error(
    y_test,
    pred_100
)

mse_100 = mean_squared_error(
    y_test,
    pred_100
)

r2_100 = r2_score(
    y_test,
    pred_100
)


# =========================================================
# 50년 / 100년 모델 기울기
# =========================================================

slope_50 = model_50.coef_[0]

slope_100 = model_100.coef_[0]

slope_50_100 = slope_50 * 100

slope_100_100 = slope_100 * 100


# =========================================================
# 회귀선 비교 그래프
# =========================================================

st.subheader(
    "50년 학습 회귀선과 100년 학습 회귀선"
)


fig_train_compare = go.Figure()


# 실제 데이터
fig_train_compare.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(size=5)
    )
)


# 50년 모델 회귀선
x50 = np.arange(
    1956,
    2026
)

y50 = model_50.predict(
    pd.DataFrame({
        "연도": x50
    })
)

fig_train_compare.add_trace(
    go.Scatter(
        x=x50,
        y=y50,
        mode="lines",
        name="50년 학습 회귀선",
        line=dict(width=3)
    )
)


# 100년 모델 회귀선
x100 = np.arange(
    1906,
    2026
)

y100 = model_100.predict(
    pd.DataFrame({
        "연도": x100
    })
)

fig_train_compare.add_trace(
    go.Scatter(
        x=x100,
        y=y100,
        mode="lines",
        name="100년 학습 회귀선",
        line=dict(width=3)
    )
)


# 테스트 기간 표시
fig_train_compare.add_vrect(
    x0=2006,
    x1=2025,
    fillcolor="gray",
    opacity=0.15,
    line_width=0,
    annotation_text="공통 테스트 2006~2025",
    annotation_position="top left"
)


fig_train_compare.update_layout(
    title="50년 학습과 100년 학습 회귀선 비교",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)


st.plotly_chart(
    fig_train_compare,
    use_container_width=True
)


# =========================================================
# 기울기 비교 표
# =========================================================

st.subheader(
    "50년 학습과 100년 학습의 기울기 비교"
)


slope_table = pd.DataFrame({
    "모델": [
        "50년 학습",
        "100년 학습"
    ],
    "학습 기간": [
        "1956~2005",
        "1906~2005"
    ],
    "테스트 기간": [
        "2006~2025",
        "2006~2025"
    ],
    "기울기 (℃/년)": [
        slope_50,
        slope_100
    ],
    "100년당 변화 (℃)": [
        slope_50_100,
        slope_100_100
    ]
})


st.dataframe(
    slope_table.style.format({
        "기울기 (℃/년)": "{:.5f}",
        "100년당 변화 (℃)": "{:+.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# MAE / MSE / R² 비교
# =========================================================

st.subheader(
    "2006~2025년 테스트 성능 비교"
)


performance_table = pd.DataFrame({
    "모델": [
        "50년 학습",
        "100년 학습"
    ],
    "학습 기간": [
        "1956~2005",
        "1906~2005"
    ],
    "테스트 기간": [
        "2006~2025",
        "2006~2025"
    ],
    "MAE (℃)": [
        mae_50,
        mae_100
    ],
    "MSE": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})


st.dataframe(
    performance_table.style.format({
        "MAE (℃)": "{:.3f}",
        "MSE": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 성능 지표를 크게 표시
# =========================================================

st.subheader(
    "50년 학습 모델"
)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "MAE",
        f"{mae_50:.3f} ℃"
    )

with c2:
    st.metric(
        "MSE",
        f"{mse_50:.3f}"
    )

with c3:
    st.metric(
        "R²",
        f"{r2_50:.3f}"
    )


st.subheader(
    "100년 학습 모델"
)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "MAE",
        f"{mae_100:.3f} ℃"
    )

with c2:
    st.metric(
        "MSE",
        f"{mse_100:.3f}"
    )

with c3:
    st.metric(
        "R²",
        f"{r2_100:.3f}"
    )


# =========================================================
# 실제값 vs 예측값
# =========================================================

st.header(
    "4. 최근 20년 실제 기온과 예측 기온 비교"
)


test_result = test_20.copy()

test_result["50년 예측"] = pred_50

test_result["100년 예측"] = pred_100


fig_test = go.Figure()


# 실제값
fig_test.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["연평균기온"],
        mode="lines+markers",
        name="실제 연평균 기온",
        line=dict(width=3)
    )
)


# 50년 예측
fig_test.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["50년 예측"],
        mode="lines+markers",
        name="50년 학습 예측",
        line=dict(
            width=2,
            dash="dash"
        )
    )
)


# 100년 예측
fig_test.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["100년 예측"],
        mode="lines+markers",
        name="100년 학습 예측",
        line=dict(
            width=2,
            dash="dot"
        )
    )
)


fig_test.update_layout(
    title="2006~2025 실제값과 두 모델의 예측값",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)


st.plotly_chart(
    fig_test,
    use_container_width=True
)


# =========================================================
# 테스트 데이터 상세표
# =========================================================

st.subheader(
    "2006~2025년 실제값과 예측값"
)


test_result["50년 오차"] = (
    test_result["연평균기온"]
    - test_result["50년 예측"]
)

test_result["100년 오차"] = (
    test_result["연평균기온"]
    - test_result["100년 예측"]
)


display_test = test_result[
    [
        "연도",
        "연평균기온",
        "50년 예측",
        "100년 예측",
        "50년 오차",
        "100년 오차"
    ]
].copy()


display_test.columns = [
    "연도",
    "실제 연평균기온",
    "50년 학습 예측",
    "100년 학습 예측",
    "50년 오차",
    "100년 오차"
]


st.dataframe(
    display_test.style.format({
        "실제 연평균기온": "{:.2f}",
        "50년 학습 예측": "{:.2f}",
        "100년 학습 예측": "{:.2f}",
        "50년 오차": "{:+.2f}",
        "100년 오차": "{:+.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 어떤 모델이 더 좋은지 자동 해석
# =========================================================

st.header(
    "5. 50년 학습과 100년 학습의 결과 해석"
)


if mae_50 < mae_100:
    mae_result = "50년 학습 모델의 MAE가 더 작습니다."
elif mae_50 > mae_100:
    mae_result = "100년 학습 모델의 MAE가 더 작습니다."
else:
    mae_result = "두 모델의 MAE가 같습니다."


if mse_50 < mse_100:
    mse_result = "50년 학습 모델의 MSE가 더 작습니다."
elif mse_50 > mse_100:
    mse_result = "100년 학습 모델의 MSE가 더 작습니다."
else:
    mse_result = "두 모델의 MSE가 같습니다."


if r2_50 > r2_100:
    r2_result = "50년 학습 모델의 R²가 더 높습니다."
elif r2_50 < r2_100:
    r2_result = "100년 학습 모델의 R²가 더 높습니다."
else:
    r2_result = "두 모델의 R²가 같습니다."


st.write(
    f"**MAE:** {mae_result}"
)

st.write(
    f"**MSE:** {mse_result}"
)

st.write(
    f"**R²:** {r2_result}"
)


# =========================================================
# 최종 요약
# =========================================================

st.subheader(
    "최종 비교"
)


final_table = pd.DataFrame({
    "항목": [
        "학습 기간",
        "테스트 기간",
        "100년당 기온 변화",
        "MAE",
        "MSE",
        "R²"
    ],
    "50년 학습": [
        "1956~2005",
        "2006~2025",
        f"{slope_50_100:+.2f} ℃",
        f"{mae_50:.3f} ℃",
        f"{mse_50:.3f}",
        f"{r2_50:.3f}"
    ],
    "100년 학습": [
        "1906~2005",
        "2006~2025",
        f"{slope_100_100:+.2f} ℃",
        f"{mae_100:.3f} ℃",
        f"{mse_100:.3f}",
        f"{r2_100:.3f}"
    ]
})


st.dataframe(
    final_table,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 지표 설명
# =========================================================

st.header(
    "6. 평가 지표 읽는 방법"
)

st.markdown(
    """
### MAE
실제 기온과 예측 기온의 차이를 절댓값으로 계산한 평균입니다.

**작을수록 좋습니다.**

### MSE
예측 오차를 제곱한 뒤 평균한 값입니다.

큰 오차에 더 큰 영향을 주며 **작을수록 좋습니다.**

### R²
모델이 실제 기온의 변동을 얼마나 설명하는지를 나타냅니다.

일반적으로 **1에 가까울수록 좋습니다.**

### 기울기
회귀선의 기울기에 100을 곱하면  
**100년에 기온이 몇 ℃ 변하는지**를 나타낼 수 있습니다.
"""
)


# =========================================================
# 데이터 테이블
# =========================================================

st.header(
    "7. 연도별 연평균 기온 데이터"
)


st.dataframe(
    annual[
        [
            "연도",
            "연평균기온",
            "전체회귀예측"
        ]
    ].style.format({
        "연평균기온": "{:.2f}",
        "전체회귀예측": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 처리 기준
# =========================================================

with st.expander("📌 데이터 처리 기준"):

    st.write(
        """
        - 서울 기온 데이터(seoul.csv)를 사용했습니다.
        - 날짜를 실제 날짜 형식으로 변환했습니다.
        - 평균기온을 숫자로 변환했습니다.
        - 2025년 이후 데이터는 제외했습니다.
        - 평균기온 관측일수가 300일 미만인 연도는 제외했습니다.
        - 남은 일별 평균기온을 연도별로 평균하여 연평균 기온을 계산했습니다.
        - 전체 기간 회귀선은 사용 가능한 전체 연평균 기온을 이용했습니다.
        - 최근 20년 회귀선은 2006~2025년 데이터를 이용했습니다.
        - 50년 학습 모델은 1956~2005년을 학습 데이터로 사용했습니다.
        - 100년 학습 모델은 1906~2005년을 학습 데이터로 사용했습니다.
        - 두 모델의 테스트 데이터는 동일하게 2006~2025년을 사용했습니다.
        - 테스트 성능은 MAE, MSE, R²로 평가했습니다.
        """
    )
