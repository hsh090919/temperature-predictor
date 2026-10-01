import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ==================================================
# 기본 설정
# ==================================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균기온 데이터를 이용해 장기간의 기온 변화 추세와 "
    "최근 20년의 변화 추세를 비교합니다."
)


# ==================================================
# 데이터 주소
# ==================================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# ==================================================
# 데이터 불러오기
# ==================================================
@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

    # 날짜
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 연도
    df["연도"] = df["날짜"].dt.year

    return df


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.code(str(e))
    st.stop()


# ==================================================
# 1. 데이터 기간 필터
# ==================================================

# 2025년까지만 사용
df = df[df["연도"] <= 2025].copy()

# 평균기온이 있는 자료만 사용
valid_temp = df.dropna(
    subset=["평균기온"]
).copy()


# ==================================================
# 2. 연도별 관측일수
# ==================================================

year_count = (
    valid_temp
    .groupby("연도")
    .size()
    .reset_index(name="관측일수")
)


# ==================================================
# 3. 연도별 평균기온
# ==================================================

year_mean = (
    valid_temp
    .groupby("연도")["평균기온"]
    .mean()
    .reset_index(name="연평균기온")
)


# ==================================================
# 4. 관측일수가 300일 이상인 해만 사용
# ==================================================

annual = pd.merge(
    year_mean,
    year_count,
    on="연도"
)

annual = annual[
    annual["관측일수"] >= 300
].copy()

annual = annual.sort_values(
    "연도"
).reset_index(drop=True)


# ==================================================
# 데이터 확인
# ==================================================

if len(annual) < 2:
    st.error(
        "회귀분석을 수행하기에 충분한 연도별 데이터가 없습니다."
    )
    st.stop()


# ==================================================
# 5. 전체 기간 회귀분석
#
# X = 1908년부터 지난 연수
# ==================================================

annual["지난연수"] = (
    annual["연도"] - 1908
)

x_all = annual["지난연수"].to_numpy(
    dtype=float
)

y_all = annual["연평균기온"].to_numpy(
    dtype=float
)


# 1차 회귀
slope_all, intercept_all = np.polyfit(
    x_all,
    y_all,
    1
)


# 1년에 몇 ℃ 변화?
slope_all_per_year = slope_all

# 100년에 몇 ℃ 변화?
slope_all_per_100 = slope_all * 100


# 상관계수
correlation_all = np.corrcoef(
    x_all,
    y_all
)[0, 1]


# ==================================================
# 6. 최근 20년 데이터
#
# 2006 ~ 2025
# ==================================================

recent_start_year = 2006
recent_end_year = 2025

recent = annual[
    (annual["연도"] >= recent_start_year)
    & (annual["연도"] <= recent_end_year)
].copy()


if len(recent) < 2:
    st.error(
        "최근 20년의 회귀분석을 수행하기에 "
        "충분한 데이터가 없습니다."
    )
    st.stop()


# 최근 20년 역시 1908년 기준 지난 연수를 X로 사용
recent["지난연수"] = (
    recent["연도"] - 1908
)

x_recent = recent["지난연수"].to_numpy(
    dtype=float
)

y_recent = recent["연평균기온"].to_numpy(
    dtype=float
)


# 최근 20년 회귀
slope_recent, intercept_recent = np.polyfit(
    x_recent,
    y_recent,
    1
)


# 100년에 몇 ℃ 변화?
slope_recent_per_100 = slope_recent * 100


# 최근 20년 상관계수
correlation_recent = np.corrcoef(
    x_recent,
    y_recent
)[0, 1]


# ==================================================
# 7. 전체 기간 정보
# ==================================================

start_year = int(
    annual["연도"].min()
)

end_year = int(
    annual["연도"].max()
)

total_years = len(annual)

recent_years_count = len(recent)


# ==================================================
# 8. 전체 기간 회귀선
# ==================================================

reg_years_all = np.arange(
    start_year,
    end_year + 1
)

reg_x_all = (
    reg_years_all - 1908
)

reg_y_all = (
    slope_all * reg_x_all
    + intercept_all
)


# ==================================================
# 9. 최근 20년 회귀선
# ==================================================

reg_years_recent = np.arange(
    recent_start_year,
    recent_end_year + 1
)

reg_x_recent = (
    reg_years_recent - 1908
)

reg_y_recent = (
    slope_recent * reg_x_recent
    + intercept_recent
)


# ==================================================
# 10. 전체 기간 기본 정보
# ==================================================

st.subheader("📊 분석에 사용된 기간")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "전체 기간",
        f"{start_year}~{end_year}"
    )

with col2:
    st.metric(
        "회귀에 사용한 해",
        f"{total_years}개"
    )

with col3:
    st.metric(
        "관측 기준",
        "연 300일 이상"
    )


# ==================================================
# 11. 핵심 결과
# ==================================================

st.subheader("🌡️ 100년에 기온이 얼마나 변했을까?")

col1, col2 = st.columns(2)


# -------------------------------
# 전체 기간
# -------------------------------
with col1:

    st.markdown(
        "### 전체 기간"
    )

    st.metric(
        "100년에 변화하는 기온",
        f"{slope_all_per_100:+.2f}℃"
    )

    st.write(
        f"{start_year}~{end_year}년 "
        f"{total_years}개 연도의 자료로 계산"
    )

    st.write(
        f"상관계수: **{correlation_all:.3f}**"
    )


# -------------------------------
# 최근 20년
# -------------------------------
with col2:

    st.markdown(
        "### 최근 20년"
    )

    st.metric(
        "100년에 변화하는 기온",
        f"{slope_recent_per_100:+.2f}℃"
    )

    st.write(
        f"{recent_start_year}~{recent_end_year}년 "
        f"{recent_years_count}개 연도의 자료로 계산"
    )

    st.write(
        f"상관계수: **{correlation_recent:.3f}**"
    )


# ==================================================
# 12. 기울기 비교 설명
# ==================================================

st.info(
    f"""
**기울기를 100년 단위로 바꾸어 표현했습니다.**

- 전체 기간: **100년에 {slope_all_per_100:+.2f}℃**
- 최근 20년: **100년에 {slope_recent_per_100:+.2f}℃**

즉, 회귀선의 원래 기울기(℃/년)에 100을 곱해서
'100년에 몇 ℃ 변화하는가'로 바꾸었습니다.
"""
)


# ==================================================
# 13. 산점도 + 두 회귀선
# ==================================================

st.subheader(
    "📈 연평균기온과 회귀선"
)

fig = go.Figure()


# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7
        ),
        customdata=np.column_stack(
            [
                annual["관측일수"],
                annual["지난연수"]
            ]
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일<br>"
            "1908년부터 지난 연수: %{customdata[1]}년"
            "<extra></extra>"
        )
    )
)


# 전체 기간 회귀선
fig.add_trace(
    go.Scatter(
        x=reg_years_all,
        y=reg_y_all,
        mode="lines",
        name=f"전체 기간 회귀선 ({slope_all_per_100:+.2f}℃/100년)",
        line=dict(
            width=4
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "전체 기간 회귀선: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 최근 20년 회귀선
fig.add_trace(
    go.Scatter(
        x=reg_years_recent,
        y=reg_y_recent,
        mode="lines",
        name=f"최근 20년 회귀선 ({slope_recent_per_100:+.2f}℃/100년)",
        line=dict(
            width=4,
            dash="dash"
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "최근 20년 회귀선: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=650,
    hovermode="x unified",
    xaxis=dict(
        showgrid=True,
        dtick=10
    ),
    yaxis=dict(
        showgrid=True
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==================================================
# 14. 회귀식
# ==================================================

st.subheader("📐 회귀식")

st.write("전체 기간 회귀식")

st.latex(
    rf"""
    \hat{{y}} =
    {intercept_all:.3f}
    +
    {slope_all:.5f}x
    """
)

st.write(
    "최근 20년 회귀식"
)

st.latex(
    rf"""
    \hat{{y}} =
    {intercept_recent:.3f}
    +
    {slope_recent:.5f}x
    """
)

st.write(
    "여기서 x는 **1908년부터 지난 연수**입니다."
)


# ==================================================
# 15. 미래 기온 예측
# ==================================================

st.subheader(
    "🔮 원하는 연도의 예상 연평균기온"
)

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# 선택한 연도
selected_x = (
    selected_year - 1908
)


# 전체 기간 회귀선으로 예측
predicted_temp = (
    slope_all * selected_x
    + intercept_all
)


st.markdown(
    f"""
    <div style="
        padding: 30px;
        border-radius: 15px;
        background-color: #f5f7fa;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    ">
        <div style="
            font-size: 24px;
            font-weight: bold;
        ">
            {selected_year}년 예상 연평균기온
        </div>

        <div style="
            font-size: 56px;
            font-weight: bold;
            margin-top: 10px;
        ">
            {predicted_temp:.2f}℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


st.write(
    f"**{selected_year}년**은 1908년을 기준으로 "
    f"**{selected_x}년 후**입니다."
)

st.caption(
    "※ 예상값은 전체 기간의 선형 회귀선을 이용한 계산값입니다. "
    "실제 미래 기온을 보장하는 예측은 아닙니다."
)


# ==================================================
# 16. 데이터 테이블
# ==================================================

with st.expander(
    "📋 회귀분석에 사용된 연도별 데이터 보기"
):

    display_df = annual[
        [
            "연도",
            "관측일수",
            "연평균기온",
            "지난연수"
        ]
    ].copy()

    display_df.columns = [
        "연도",
        "관측일수",
        "연평균기온(℃)",
        "1908년부터 지난 연수"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# 17. 데이터 처리 기준
# ==================================================

with st.expander(
    "ℹ️ 데이터 처리 기준"
):

    st.write(
        """
        - 서울 기온 원본 데이터를 사용했습니다.
        - 2025년 이후 데이터는 제외했습니다.
        - 평균기온이 기록된 날짜가 300일 미만인 연도는 제외했습니다.
        - 남은 일별 평균기온을 연도별로 평균내어 연평균기온을 계산했습니다.
        - 회귀분석의 독립변수는 `연도 - 1908`입니다.
        - 전체 기간 회귀선과 2006~2025년 최근 20년 회귀선을 각각 계산했습니다.
        - 기울기는 ℃/년에서 ℃/100년으로 변환하여 표시했습니다.
        - 슬라이더의 범위는 1900~2100년입니다.
        """
    )


