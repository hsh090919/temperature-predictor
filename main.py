import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 선형회귀 분석")

st.write(
    "서울 연평균 기온 데이터를 이용하여 선형회귀 모델을 만들고 "
    "전체 데이터와 50년·100년 학습 모델의 예측 성능을 비교합니다."
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

required_columns = ["날짜", "평균기온"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        f"데이터에 다음 열이 없습니다: {missing_columns}"
    )

    st.write("현재 데이터의 열:")
    st.write(df.columns.tolist())

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
# 선형회귀 함수
# numpy만 사용
# =========================================================

def make_regression(data):

    x = data["연도"].to_numpy(dtype=float)
    y = data["연평균기온"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    return slope, intercept


def predict(slope, intercept, years):

    years = np.asarray(
        years,
        dtype=float
    )

    return (
        slope * years
        + intercept
    )


# =========================================================
# 평가 지표
# =========================================================

def calculate_mae(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    return np.mean(
        np.abs(
            y_true - y_pred
        )
    )


def calculate_mse(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    return np.mean(
        (y_true - y_pred) ** 2
    )


def calculate_r2(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot == 0:
        return np.nan

    return 1 - (
        ss_res / ss_tot
    )


# =========================================================
# 전체 데이터 회귀
# =========================================================

full_slope, full_intercept = make_regression(
    annual
)

annual["전체회귀예측"] = predict(
    full_slope,
    full_intercept,
    annual["연도"]
)


# =========================================================
# 전체 데이터 평가
# =========================================================

full_mae = calculate_mae(
    annual["연평균기온"],
    annual["전체회귀예측"]
)

full_mse = calculate_mse(
    annual["연평균기온"],
    annual["전체회귀예측"]
)

full_r2 = calculate_r2(
    annual["연평균기온"],
    annual["전체회귀예측"]
)

full_slope_100 = full_slope * 100


# =========================================================
# ① 전체 데이터 평가
# =========================================================

st.header("① 전체 데이터에 대한 선형회귀 평가")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "기울기",
        f"{full_slope:.4f} ℃/년"
    )

with col2:
    st.metric(
        "100년당 변화",
        f"{full_slope_100:+.2f} ℃"
    )

with col3:
    st.metric(
        "MAE",
        f"{full_mae:.3f} ℃"
    )

with col4:
    st.metric(
        "R²",
        f"{full_r2:.3f}"
    )


st.write(
    f"**전체 데이터 회귀식:** "
    f"연평균기온 = {full_slope:.4f} × 연도 + {full_intercept:.2f}"
)


# =========================================================
# 전체 데이터 그래프
# =========================================================

fig_full = go.Figure()

fig_full.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(size=5)
    )
)

fig_full.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["전체회귀예측"],
        mode="lines",
        name="전체 데이터 회귀선",
        line=dict(width=3)
    )
)

fig_full.update_layout(
    title="전체 기간의 연평균 기온과 선형회귀선",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)

st.plotly_chart(
    fig_full,
    use_container_width=True
)


# =========================================================
# ② 훈련 / 테스트 데이터 분리
# =========================================================

st.header("② 훈련 데이터와 테스트 데이터 분리")

st.write(
    "과거 연도를 훈련 데이터로 사용하고, "
    "최근 20년(2006~2025)을 두 모델의 공통 테스트 데이터로 사용합니다."
)


# 50년 학습
train_50 = annual[
    (annual["연도"] >= 1956) &
    (annual["연도"] <= 2005)
].copy()


# 100년 학습
train_100 = annual[
    (annual["연도"] >= 1906) &
    (annual["연도"] <= 2005)
].copy()


# 공통 테스트
test = annual[
    (annual["연도"] >= 2006) &
    (annual["연도"] <= 2025)
].copy()


if train_50.empty:
    st.error("1956~2005년 학습 데이터가 없습니다.")
    st.stop()

if train_100.empty:
    st.error("1906~2005년 학습 데이터가 없습니다.")
    st.stop()

if test.empty:
    st.error("2006~2025년 테스트 데이터가 없습니다.")
    st.stop()


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "50년 학습 데이터",
        f"{len(train_50)}개 연도"
    )

with col2:
    st.metric(
        "100년 학습 데이터",
        f"{len(train_100)}개 연도"
    )

with col3:
    st.metric(
        "공통 테스트 데이터",
        f"{len(test)}개 연도"
    )


st.markdown(
    """
| 구분 | 기간 |
|---|---|
| 50년 학습 | **1956~2005** |
| 100년 학습 | **1906~2005** |
| 공통 테스트 | **2006~2025** |
"""
)


# =========================================================
# 50년 모델
# =========================================================

slope_50, intercept_50 = make_regression(
    train_50
)

pred_50 = predict(
    slope_50,
    intercept_50,
    test["연도"]
)


# =========================================================
# 100년 모델
# =========================================================

slope_100, intercept_100 = make_regression(
    train_100
)

pred_100 = predict(
    slope_100,
    intercept_100,
    test["연도"]
)


# =========================================================
# 테스트 성능 계산
# =========================================================

y_test = test[
    "연평균기온"
].to_numpy()


# 50년 모델 평가

mae_50 = calculate_mae(
    y_test,
    pred_50
)

mse_50 = calculate_mse(
    y_test,
    pred_50
)

r2_50 = calculate_r2(
    y_test,
    pred_50
)


# 100년 모델 평가

mae_100 = calculate_mae(
    y_test,
    pred_100
)

mse_100 = calculate_mse(
    y_test,
    pred_100
)

r2_100 = calculate_r2(
    y_test,
    pred_100
)


# =========================================================
# 100년당 기울기
# =========================================================

slope_50_100 = slope_50 * 100
slope_100_100 = slope_100 * 100


# =========================================================
# ③ 회귀선 비교
# =========================================================

st.header("③ 50년 학습과 100년 학습의 회귀선 비교")


st.subheader("회귀선의 기울기 비교")


slope_table = pd.DataFrame({
    "모델": [
        "50년 학습 모델",
        "100년 학습 모델"
    ],
    "학습 기간": [
        "1956~2005",
        "1906~2005"
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
# 회귀선 그래프
# =========================================================

x_50 = np.arange(
    1956,
    2026
)

y_50 = predict(
    slope_50,
    intercept_50,
    x_50
)


x_100 = np.arange(
    1906,
    2026
)

y_100 = predict(
    slope_100,
    intercept_100,
    x_100
)


fig_compare = go.Figure()


fig_compare.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(size=5)
    )
)


fig_compare.add_trace(
    go.Scatter(
        x=x_50,
        y=y_50,
        mode="lines",
        name="50년 학습 회귀선",
        line=dict(width=3)
    )
)


fig_compare.add_trace(
    go.Scatter(
        x=x_100,
        y=y_100,
        mode="lines",
        name="100년 학습 회귀선",
        line=dict(width=3)
    )
)


fig_compare.add_vrect(
    x0=2006,
    x1=2025,
    fillcolor="gray",
    opacity=0.15,
    line_width=0,
    annotation_text="공통 테스트 기간",
    annotation_position="top left"
)


fig_compare.update_layout(
    title="50년 학습 회귀선과 100년 학습 회귀선",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)


st.plotly_chart(
    fig_compare,
    use_container_width=True
)


# =========================================================
# ④ 테스트 데이터 예측 성능
# =========================================================

st.header("④ 테스트 데이터 예측 성능")

st.write(
    "두 모델이 학습에 사용하지 않은 "
    "**최근 20년(2006~2025)**을 얼마나 잘 예측했는지 비교합니다."
)


# =========================================================
# 50년 / 100년 모델 성능을 좌우로 표시
# =========================================================

left, right = st.columns(2)


# ---------------------------------------------------------
# 50년 학습 모델
# ---------------------------------------------------------

with left:

    st.subheader("🔷 50년 학습 모델")

    st.caption("학습: 1956~2005 / 테스트: 2006~2025")

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


# ---------------------------------------------------------
# 100년 학습 모델
# ---------------------------------------------------------

with right:

    st.subheader("🔶 100년 학습 모델")

    st.caption("학습: 1906~2005 / 테스트: 2006~2025")

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
# 모델 비교표
# =========================================================

st.subheader("50년 학습 모델과 100년 학습 모델 비교")


performance_table = pd.DataFrame({
    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],
    "학습 데이터": [
        "1956~2005",
        "1906~2005"
    ],
    "테스트 데이터": [
        "2006~2025",
        "2006~2025"
    ],
    "학습 데이터 수": [
        len(train_50),
        len(train_100)
    ],
    "기울기 (℃/년)": [
        slope_50,
        slope_100
    ],
    "100년당 변화 (℃)": [
        slope_50_100,
        slope_100_100
    ],
    "MAE": [
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
        "기울기 (℃/년)": "{:.5f}",
        "100년당 변화 (℃)": "{:+.2f}",
        "MAE": "{:.3f}",
        "MSE": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# ⑤ 실제값 vs 예측값
# =========================================================

st.header("⑤ 최근 20년 실제값과 예측값 비교")


test_result = test[
    ["연도", "연평균기온"]
].copy()


test_result["50년 예측"] = pred_50

test_result["100년 예측"] = pred_100

test_result["50년 오차"] = (
    test_result["연평균기온"]
    - test_result["50년 예측"]
)

test_result["100년 오차"] = (
    test_result["연평균기온"]
    - test_result["100년 예측"]
)


fig_test = go.Figure()


fig_test.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["연평균기온"],
        mode="lines+markers",
        name="실제 연평균 기온",
        line=dict(width=3)
    )
)


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
    title="2006~2025 실제 기온과 두 모델의 예측",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)


st.plotly_chart(
    fig_test,
    use_container_width=True
)


# =========================================================
# ⑥ 결과 해석
# =========================================================

st.header("⑥ 50년 학습과 100년 학습의 결과 해석")


if mae_50 < mae_100:
    mae_result = "50년 학습 모델의 MAE가 더 작아 예측 오차가 더 작습니다."
elif mae_50 > mae_100:
    mae_result = "100년 학습 모델의 MAE가 더 작아 예측 오차가 더 작습니다."
else:
    mae_result = "두 모델의 MAE가 같습니다."


if mse_50 < mse_100:
    mse_result = "50년 학습 모델의 MSE가 더 작습니다."
elif mse_50 > mse_100:
    mse_result = "100년 학습 모델의 MSE가 더 작습니다."
else:
    mse_result = "두 모델의 MSE가 같습니다."


if r2_50 > r2_100:
    r2_result = "50년 학습 모델의 R²가 더 높아 테스트 데이터의 변동을 더 잘 설명합니다."
elif r2_50 < r2_100:
    r2_result = "100년 학습 모델의 R²가 더 높아 테스트 데이터의 변동을 더 잘 설명합니다."
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
# ⑦ 연평균 기온 데이터
# =========================================================

st.header("⑦ 연도별 연평균 기온")


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
# ⑧ 평가 지표 설명
# =========================================================

st.header("⑧ 평가 지표 읽는 방법")

st.markdown(
    """
### MAE
실제 기온과 예측 기온의 차이를 절댓값으로 계산한 평균입니다.

**작을수록 예측이 좋습니다.**

### MSE
실제값과 예측값의 차이를 제곱한 뒤 평균한 값입니다.

큰 오차에 더 큰 영향을 주며 **작을수록 좋습니다.**

### R²
모델이 테스트 데이터의 기온 변동을 얼마나 설명하는지를 나타냅니다.

일반적으로 **1에 가까울수록 좋습니다.**

### 기울기
회귀선의 기울기에 100을 곱하면

**100년에 기온이 몇 ℃ 변하는가**

를 나타냅니다.
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
        - 전체 데이터 회귀모델은 전체 연평균 기온을 이용했습니다.
        - 50년 학습 모델은 1956~2005년을 훈련 데이터로 사용했습니다.
        - 100년 학습 모델은 1906~2005년을 훈련 데이터로 사용했습니다.
        - 두 모델의 테스트 데이터는 동일하게 2006~2025년을 사용했습니다.
        - 테스트 성능은 MAE, MSE, R²로 평가했습니다.
        - 선형회귀와 평가 지표 계산에는 numpy를 사용했습니다.
        - scikit-learn은 사용하지 않습니다.
        """
    )
