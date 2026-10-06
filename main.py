import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from numpy.polynomial import Polynomial


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 분석 및 예측")

st.write(
    "서울 연평균 기온 데이터를 이용하여 선형회귀와 "
    "다항회귀 모델의 예측 성능을 비교합니다."
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
# 데이터 열 확인
# =========================================================

if "날짜" not in df.columns or "평균기온" not in df.columns:
    st.error("데이터에 '날짜' 또는 '평균기온' 열이 없습니다.")
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
# 1906~2025년 데이터 사용
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
# =========================================================

def make_linear_regression(data):

    x = data["연도"].to_numpy(
        dtype=float
    )

    y = data["연평균기온"].to_numpy(
        dtype=float
    )

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    return slope, intercept


def predict_linear(
    slope,
    intercept,
    years
):

    years = np.asarray(
        years,
        dtype=float
    )

    return (
        slope * years
        + intercept
    )


# =========================================================
# 평가 함수
# =========================================================

def calculate_mae(
    y_true,
    y_pred
):

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


def calculate_mse(
    y_true,
    y_pred
):

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


def calculate_r2(
    y_true,
    y_pred
):

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
# 1. 전체 데이터 선형회귀
# =========================================================

st.header("① 전체 데이터에 대한 선형회귀 평가")

full_slope, full_intercept = make_linear_regression(
    annual
)

annual["전체회귀예측"] = predict_linear(
    full_slope,
    full_intercept,
    annual["연도"]
)


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


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "회귀선 기울기",
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


st.info(
    "전체 데이터의 MAE와 R²는 회귀선을 만든 동일한 데이터로 평가한 값입니다. "
    "학습에 사용하지 않은 데이터의 예측 성능은 아래 테스트 평가에서 확인합니다."
)


# =========================================================
# 2. 50년 / 100년 학습 데이터 분리
# =========================================================

st.header("② 50년 학습과 100년 학습 데이터 분리")

st.write(
    "두 모델 모두 2006~2025년을 공통 테스트 데이터로 사용합니다."
)


train_50 = annual[
    (annual["연도"] >= 1956) &
    (annual["연도"] <= 2005)
].copy()


train_100 = annual[
    (annual["연도"] >= 1906) &
    (annual["연도"] <= 2005)
].copy()


test_20 = annual[
    (annual["연도"] >= 2006) &
    (annual["연도"] <= 2025)
].copy()


if train_50.empty:
    st.error("1956~2005년 학습 데이터가 없습니다.")
    st.stop()

if train_100.empty:
    st.error("1906~2005년 학습 데이터가 없습니다.")
    st.stop()

if test_20.empty:
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
        f"{len(test_20)}개 연도"
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
# 50년 / 100년 모델 학습
# =========================================================

slope_50, intercept_50 = make_linear_regression(
    train_50
)

pred_50 = predict_linear(
    slope_50,
    intercept_50,
    test_20["연도"]
)


slope_100, intercept_100 = make_linear_regression(
    train_100
)

pred_100 = predict_linear(
    slope_100,
    intercept_100,
    test_20["연도"]
)


y_test_20 = test_20[
    "연평균기온"
].to_numpy()


mae_50 = calculate_mae(
    y_test_20,
    pred_50
)

mse_50 = calculate_mse(
    y_test_20,
    pred_50
)

r2_50 = calculate_r2(
    y_test_20,
    pred_50
)


mae_100 = calculate_mae(
    y_test_20,
    pred_100
)

mse_100 = calculate_mse(
    y_test_20,
    pred_100
)

r2_100 = calculate_r2(
    y_test_20,
    pred_100
)


slope_50_100 = slope_50 * 100
slope_100_100 = slope_100 * 100


# =========================================================
# 3. 50년 / 100년 회귀선 비교
# =========================================================

st.header("③ 50년 학습과 100년 학습의 회귀선 비교")


slope_table = pd.DataFrame({
    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],
    "학습 기간": [
        "1956~2005",
        "1906~2005"
    ],
    "기울기 (℃/년)": [
        slope_50,
        slope_100
    ],
    "100년당 기온 변화 (℃)": [
        slope_50_100,
        slope_100_100
    ]
})


st.dataframe(
    slope_table.style.format({
        "기울기 (℃/년)": "{:.5f}",
        "100년당 기온 변화 (℃)": "{:+.2f}"
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

y_50 = predict_linear(
    slope_50,
    intercept_50,
    x_50
)


x_100 = np.arange(
    1906,
    2026
)

y_100 = predict_linear(
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
# 4. 테스트 데이터 예측 성능
# =========================================================

st.header("④ 테스트 데이터 예측 성능")

st.write(
    "50년·100년 모델은 학습에 사용하지 않은 "
    "**2006~2025년 데이터**로 평가합니다."
)


left, right = st.columns(2)


with left:

    st.subheader("🔷 50년 학습 모델")

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


with right:

    st.subheader("🔶 100년 학습 모델")

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
# 50년 / 100년 비교표
# =========================================================

st.subheader("50년 학습 모델과 100년 학습 모델 비교")


performance = pd.DataFrame({
    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
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
    performance.style.format({
        "MAE (℃)": "{:.3f}",
        "MSE": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 5. 실제값과 50년 / 100년 예측값
# =========================================================

st.header("⑤ 2006~2025 실제값과 예측값 비교")


test_result_20 = test_20[
    ["연도", "연평균기온"]
].copy()


test_result_20["50년 예측"] = pred_50

test_result_20["100년 예측"] = pred_100

test_result_20["50년 오차"] = (
    test_result_20["연평균기온"]
    - test_result_20["50년 예측"]
)

test_result_20["100년 오차"] = (
    test_result_20["연평균기온"]
    - test_result_20["100년 예측"]
)


fig_test_20 = go.Figure()


fig_test_20.add_trace(
    go.Scatter(
        x=test_result_20["연도"],
        y=test_result_20["연평균기온"],
        mode="lines+markers",
        name="실제 연평균 기온",
        line=dict(width=3)
    )
)


fig_test_20.add_trace(
    go.Scatter(
        x=test_result_20["연도"],
        y=test_result_20["50년 예측"],
        mode="lines+markers",
        name="50년 학습 예측",
        line=dict(
            width=2,
            dash="dash"
        )
    )
)


fig_test_20.add_trace(
    go.Scatter(
        x=test_result_20["연도"],
        y=test_result_20["100년 예측"],
        mode="lines+markers",
        name="100년 학습 예측",
        line=dict(
            width=2,
            dash="dot"
        )
    )
)


fig_test_20.update_layout(
    title="2006~2025 실제 기온과 두 회귀모델의 예측",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)


st.plotly_chart(
    fig_test_20,
    use_container_width=True
)


# =========================================================
# 6. 50년 / 100년 모델 결과 해석
# =========================================================

st.header("⑥ 50년 학습과 100년 학습의 비교 결과")


if mae_50 < mae_100:
    mae_result = "50년 학습 모델이 더 좋습니다."
elif mae_50 > mae_100:
    mae_result = "100년 학습 모델이 더 좋습니다."
else:
    mae_result = "두 모델의 성능이 같습니다."


if mse_50 < mse_100:
    mse_result = "50년 학습 모델이 더 좋습니다."
elif mse_50 > mse_100:
    mse_result = "100년 학습 모델이 더 좋습니다."
else:
    mse_result = "두 모델의 성능이 같습니다."


if r2_50 > r2_100:
    r2_result = "50년 학습 모델이 더 좋습니다."
elif r2_50 < r2_100:
    r2_result = "100년 학습 모델이 더 좋습니다."
else:
    r2_result = "두 모델의 성능이 같습니다."


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
# 7. 2005년 기준 훈련 / 테스트 분리
# =========================================================

st.header("⑦ 1차·3차·9차 곡선을 위한 훈련 / 테스트 분리")


st.write(
    "이번에는 2005년을 기준으로 데이터를 다시 나눕니다."
)


# ---------------------------------------------------------
# 2005년 이전 = 훈련
# 2005년부터 = 테스트
# ---------------------------------------------------------

train_curve = annual[
    annual["연도"] < 2005
].copy()


test_curve = annual[
    annual["연도"] >= 2005
].copy()


if train_curve.empty:
    st.error("곡선 모델의 훈련 데이터가 없습니다.")
    st.stop()


if test_curve.empty:
    st.error("곡선 모델의 테스트 데이터가 없습니다.")
    st.stop()


# =========================================================
# 훈련 / 테스트 개수 표시
# =========================================================

col1, col2 = st.columns(2)


with col1:

    st.metric(
        "훈련용 연도",
        f"{len(train_curve)}개"
    )

    st.caption(
        f"{train_curve['연도'].min()}~"
        f"{train_curve['연도'].max()}년"
    )


with col2:

    st.metric(
        "테스트용 연도",
        f"{len(test_curve)}개"
    )

    st.caption(
        f"{test_curve['연도'].min()}~"
        f"{test_curve['연도'].max()}년"
    )


split_curve_table = pd.DataFrame({
    "구분": [
        "훈련용",
        "테스트용"
    ],
    "기간": [
        f"{train_curve['연도'].min()}~"
        f"{train_curve['연도'].max()}",
        f"{test_curve['연도'].min()}~"
        f"{test_curve['연도'].max()}"
    ],
    "연도 개수": [
        len(train_curve),
        len(test_curve)
    ],
    "모델 학습에 사용": [
        "O",
        "X"
    ]
})


st.dataframe(
    split_curve_table,
    use_container_width=True,
    hide_index=True
)


st.info(
    "중요: 테스트용 데이터인 2005년 이후 데이터는 "
    "1차·3차·9차 모델을 학습할 때 사용하지 않습니다."
)


# =========================================================
# 8. 연도 숫자 안정화
# =========================================================

st.header("⑧ 고차 곡선 계산을 위한 연도 변환")


st.write(
    "9차 곡선처럼 높은 차수의 다항식을 계산할 때 "
    "1906, 1907 같은 큰 숫자를 그대로 거듭제곱하면 "
    "계산이 불안정해질 수 있습니다."
)

st.write(
    "그래서 계산할 때만 **연도 - 2005**로 바꿉니다."
)

st.markdown(
    """
| 실제 연도 | 계산용 연도 |
|---:|---:|
| 1906 | -99 |
| 1950 | -55 |
| 2004 | -1 |
| 2005 | 0 |
| 2025 | 20 |
| 2050 | 45 |
"""
)


# =========================================================
# 계산용 X, Y
# =========================================================

train_x_curve = (
    train_curve["연도"].to_numpy(
        dtype=float
    )
    - 2005
)

train_y_curve = train_curve[
    "연평균기온"
].to_numpy(
    dtype=float
)


test_x_curve = (
    test_curve["연도"].to_numpy(
        dtype=float
    )
    - 2005
)

test_y_curve = test_curve[
    "연평균기온"
].to_numpy(
    dtype=float
)


# =========================================================
# 9. 1차 / 3차 / 9차 곡선 학습
# =========================================================

st.header("⑨ 1차·3차·9차 곡선 학습")


# ---------------------------------------------------------
# 반드시 훈련 데이터만 사용
# ---------------------------------------------------------

model_1 = Polynomial.fit(
    train_x_curve,
    train_y_curve,
    1
)


model_3 = Polynomial.fit(
    train_x_curve,
    train_y_curve,
    3
)


model_9 = Polynomial.fit(
    train_x_curve,
    train_y_curve,
    9
)


st.success(
    "1차·3차·9차 모델의 학습을 완료했습니다. "
    "세 모델 모두 2005년 이전 훈련 데이터만 사용했습니다."
)


# =========================================================
# 10. 테스트 데이터 예측
# =========================================================

st.header("⑩ 학습에 사용하지 않은 테스트 데이터로 채점")


# ---------------------------------------------------------
# 테스트 데이터는 여기에서 처음 사용
# ---------------------------------------------------------

pred_1_test = model_1(
    test_x_curve
)

pred_3_test = model_3(
    test_x_curve
)

pred_9_test = model_9(
    test_x_curve
)


# =========================================================
# 테스트 MAE
# =========================================================

mae_curve_1 = calculate_mae(
    test_y_curve,
    pred_1_test
)

mae_curve_3 = calculate_mae(
    test_y_curve,
    pred_3_test
)

mae_curve_9 = calculate_mae(
    test_y_curve,
    pred_9_test
)


# =========================================================
# 2050년 예측
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
# 11. 테스트 성능 카드
# =========================================================

st.header("⑪ 테스트 데이터 예측 성능")


left, middle, right = st.columns(3)


with left:

    st.subheader("1차(직선)")

    st.metric(
        "테스트 MAE",
        f"{mae_curve_1:.3f} ℃"
    )


with middle:

    st.subheader("3차 곡선")

    st.metric(
        "테스트 MAE",
        f"{mae_curve_3:.3f} ℃"
    )


with right:

    st.subheader("9차 곡선")

    st.metric(
        "테스트 MAE",
        f"{mae_curve_9:.3f} ℃"
    )


# =========================================================
# 12. 최종 비교표
# =========================================================

st.header("⑫ 곡선별 테스트 성능과 2050년 예측")


curve_result = pd.DataFrame({
    "모델": [
        "1차(직선)",
        "3차 곡선",
        "9차 곡선"
    ],
    "훈련 기간": [
        f"{train_curve['연도'].min()}~"
        f"{train_curve['연도'].max()}",
        f"{train_curve['연도'].min()}~"
        f"{train_curve['연도'].max()}",
        f"{train_curve['연도'].min()}~"
        f"{train_curve['연도'].max()}"
    ],
    "테스트 기간": [
        f"{test_curve['연도'].min()}~"
        f"{test_curve['연도'].max()}",
        f"{test_curve['연도'].min()}~"
        f"{test_curve['연도'].max()}",
        f"{test_curve['연도'].min()}~"
        f"{test_curve['연도'].max()}"
    ],
    "테스트 MAE (℃)": [
        mae_curve_1,
        mae_curve_3,
        mae_curve_9
    ],
    "2050년 예측 (℃)": [
        prediction_2050_1,
        prediction_2050_3,
        prediction_2050_9
    ]
})


st.dataframe(
    curve_result.style.format({
        "테스트 MAE (℃)": "{:.3f}",
        "2050년 예측 (℃)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 13. 1차·3차·9차 곡선 그래프
# =========================================================

st.header("⑬ 1차·3차·9차 곡선 비교")


graph_years = np.linspace(
    1906,
    2050,
    500
)


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


fig_curve = go.Figure()


# 훈련 데이터
fig_curve.add_trace(
    go.Scatter(
        x=train_curve["연도"],
        y=train_curve["연평균기온"],
        mode="markers",
        name="훈련용 실제값",
        marker=dict(size=5)
    )
)


# 테스트 데이터
fig_curve.add_trace(
    go.Scatter(
        x=test_curve["연도"],
        y=test_curve["연평균기온"],
        mode="markers",
        name="테스트용 실제값",
        marker=dict(size=6)
    )
)


# 1차
fig_curve.add_trace(
    go.Scatter(
        x=graph_years,
        y=graph_pred_1,
        mode="lines",
        name="1차 곡선(직선)",
        line=dict(width=3)
    )
)


# 3차
fig_curve.add_trace(
    go.Scatter(
        x=graph_years,
        y=graph_pred_3,
        mode="lines",
        name="3차 곡선",
        line=dict(width=3)
    )
)


# 9차
fig_curve.add_trace(
    go.Scatter(
        x=graph_years,
        y=graph_pred_9,
        mode="lines",
        name="9차 곡선",
        line=dict(width=3)
    )
)


# 2005년 기준선
fig_curve.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="훈련 → 테스트",
    annotation_position="top"
)


# 2050년 기준선
fig_curve.add_vline(
    x=2050,
    line_dash="dot",
    annotation_text="2050년",
    annotation_position="top"
)


fig_curve.update_layout(
    title="훈련 데이터로만 학습한 1차·3차·9차 곡선",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)


st.plotly_chart(
    fig_curve,
    use_container_width=True
)


# =========================================================
# 14. 테스트 실제값과 예측값 비교
# =========================================================

st.header("⑭ 테스트 데이터의 실제값과 예측값")


curve_test_result = test_curve[
    ["연도", "연평균기온"]
].copy()


curve_test_result["1차 예측"] = pred_1_test

curve_test_result["3차 예측"] = pred_3_test

curve_test_result["9차 예측"] = pred_9_test


fig_curve_test = go.Figure()


fig_curve_test.add_trace(
    go.Scatter(
        x=curve_test_result["연도"],
        y=curve_test_result["연평균기온"],
        mode="lines+markers",
        name="실제 기온",
        line=dict(width=3)
    )
)


fig_curve_test.add_trace(
    go.Scatter(
        x=curve_test_result["연도"],
        y=curve_test_result["1차 예측"],
        mode="lines",
        name="1차 예측",
        line=dict(
            width=2,
            dash="dash"
        )
    )
)


fig_curve_test.add_trace(
    go.Scatter(
        x=curve_test_result["연도"],
        y=curve_test_result["3차 예측"],
        mode="lines",
        name="3차 예측",
        line=dict(
            width=2,
            dash="dot"
        )
    )
)


fig_curve_test.add_trace(
    go.Scatter(
        x=curve_test_result["연도"],
        y=curve_test_result["9차 예측"],
        mode="lines",
        name="9차 예측",
        line=dict(
            width=2,
            dash="longdash"
        )
    )
)


fig_curve_test.update_layout(
    title="학습에 사용하지 않은 테스트 데이터의 실제값과 예측값",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)


st.plotly_chart(
    fig_curve_test,
    use_container_width=True
)


# =========================================================
# 15. 가장 좋은 곡선
# =========================================================

st.header("⑮ 테스트 결과 해석")


curve_mae_values = {
    "1차(직선)": mae_curve_1,
    "3차 곡선": mae_curve_3,
    "9차 곡선": mae_curve_9
}


best_curve = min(
    curve_mae_values,
    key=curve_mae_values.get
)


st.success(
    f"테스트 데이터의 MAE가 가장 작은 모델은 "
    f"**{best_curve}**입니다. "
    f"테스트 데이터에서 평균적으로 약 "
    f"**{curve_mae_values[best_curve]:.3f}℃** "
    f"정도 빗나갔습니다."
)


st.write(
    "MAE는 실제값과 예측값의 차이를 절댓값으로 계산한 평균이므로 "
    "**작을수록 테스트 데이터에 대한 예측 성능이 좋습니다.**"
)


# =========================================================
# 16. 2050년 예측
# =========================================================

st.header("⑯ 2050년 기온 예측")


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
# 17. 연도별 연평균 기온
# =========================================================

st.header("⑰ 연도별 연평균 기온")


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
# 18. 평가 지표 설명
# =========================================================

st.header("⑱ 평가 지표 읽는 방법")

st.markdown(
    """
### MAE

실제 기온과 예측 기온의 차이를 절댓값으로 계산한 평균입니다.

**작을수록 좋습니다.**

---

### MSE

실제값과 예측값의 차이를 제곱한 뒤 평균한 값입니다.

큰 오차에 더 큰 영향을 주며 **작을수록 좋습니다.**

---

### R²

모델이 실제 기온의 변동을 얼마나 설명하는지를 나타냅니다.

일반적으로 **1에 가까울수록 좋습니다.**

---

### 기울기

선형회귀선의 기울기에 100을 곱하면

**"100년에 기온이 몇 ℃ 변하는가"**

로 해석할 수 있습니다.

---

### 다항식 차수

**1차**는 직선입니다.

**3차**는 더 유연하게 휘어지는 곡선입니다.

**9차**는 매우 복잡한 곡선을 만들 수 있습니다.

하지만 차수가 높다고 반드시 미래 예측을 잘하는 것은 아닙니다.

그래서 이번 분석에서는 반드시 **학습에 사용하지 않은 테스트 데이터**로 성능을 평가합니다.
"""
)


# =========================================================
# 19. 데이터 처리 기준
# =========================================================

with st.expander("📌 데이터 처리 기준"):

    st.write(
        """
        [기본 선형회귀 분석]

        - 서울 기온 데이터(seoul.csv)를 사용했습니다.
        - 날짜를 날짜 형식으로 변환했습니다.
        - 평균기온을 숫자로 변환했습니다.
        - 1906~2025년 데이터를 사용했습니다.
        - 일별 평균기온을 연도별로 평균하여 연평균 기온을 계산했습니다.
        - 전체 데이터 회귀모델은 전체 연평균 기온을 이용했습니다.
        - 50년 학습 모델은 1956~2005년을 훈련 데이터로 사용했습니다.
        - 100년 학습 모델은 1906~2005년을 훈련 데이터로 사용했습니다.
        - 50년·100년 모델의 공통 테스트 데이터는 2006~2025년입니다.

        [다항회귀 분석]

        - 2005년 이전(2004년까지)을 훈련 데이터로 사용했습니다.
        - 2005년부터를 테스트 데이터로 사용했습니다.
        - 1차·3차·9차 모델 모두 훈련 데이터만 사용하여 학습했습니다.
        - 테스트 데이터는 모델 학습에 사용하지 않았습니다.
        - 테스트 데이터에서 MAE를 계산했습니다.
        - 각 모델에 2050년을 입력하여 2050년 기온을 예측했습니다.
        - 고차 다항식의 계산 안정성을 위해
          실제 연도 대신 '연도 - 2005'를 계산에 사용했습니다.
        - 다항식 계산에는 numpy를 사용했습니다.
        - scikit-learn은 사용하지 않습니다.
        """
    )
