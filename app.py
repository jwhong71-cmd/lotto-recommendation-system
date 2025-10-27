"""
로또 6/45 분석 및 추천 시스템 - Streamlit 웹앱 버전
동행복권 API를 통한 데이터 수집 및 7가지 알고리즘 기반 번호 추천
Edward Thorp (켈리 공식)와 Blaise Pascal (조합론) 알고리즘 추가
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from collections import defaultdict, Counter
import itertools
import random
import time
import json
import io
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# 페이지 설정
st.set_page_config(
    page_title="🎯 로또 분석 추천 시스템 v2.0",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 동행복권 API 엔드포인트
API = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={}"

# ---------- 최신 회차 계산 ----------
@st.cache_data(ttl=3600)  # 1시간 캐시
def get_latest_round():
    """현재 날짜를 기준으로 최신 로또 회차를 계산"""
    first_draw_date = datetime(2002, 12, 7)
    current_date = datetime.now()
    
    days_since_saturday = (current_date.weekday() + 2) % 7
    current_saturday = current_date - timedelta(days=days_since_saturday)
    draw_time = current_saturday.replace(hour=20, minute=45, second=0, microsecond=0)
    
    if current_date < draw_time:
        current_saturday -= timedelta(days=7)
    
    weeks_diff = (current_saturday - first_draw_date).days // 7
    latest_round = weeks_diff + 1
    
    return max(1, min(latest_round, 2000))

# ---------- 데이터 수집 함수 ----------
@st.cache_data(ttl=3600)  # 1시간 캐시 (더 긴 캐시)
def fetch_lotto_data(start_round, end_round):
    """로또 데이터 수집 - 성능 최적화 버전"""
    data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_rounds = end_round - start_round + 1
    
    # 배치 처리를 위한 설정
    batch_size = 10  # 10개씩 병렬 처리
    failed_rounds = []
    

    
    def fetch_single_round(round_num):
        """단일 회차 데이터 수집"""
        try:
            response = requests.get(API.format(round_num), timeout=5)  # 타임아웃 단축
            if response.status_code == 200:
                result = response.json()
                if result.get('returnValue') == 'success':
                    winning_numbers = [result[f'drwtNo{j}'] for j in range(1, 7)]
                    return {
                        'round': round_num,
                        'numbers': winning_numbers,
                        'bonus': result['bnusNo'],
                        'date': result['drwNoDate']
                    }
        except Exception:
            return None
        return None
    
    # 병렬 처리로 속도 개선
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        rounds = list(range(start_round, end_round + 1))
        
        for i in range(0, len(rounds), batch_size):
            batch = rounds[i:i + batch_size]
            
            # 진행률 업데이트
            progress = (i + len(batch)) / total_rounds
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"고속 데이터 수집 중... {i+1}~{min(i+batch_size, len(rounds))}회차 ({i+len(batch)}/{total_rounds})")
            
            # 배치 병렬 처리
            future_to_round = {executor.submit(fetch_single_round, round_num): round_num for round_num in batch}
            
            for future in concurrent.futures.as_completed(future_to_round):
                result = future.result()
                if result:
                    data.append(result)
                else:
                    failed_rounds.append(future_to_round[future])
            
            # API 부하 방지를 위한 짧은 대기
            time.sleep(0.05)
    
    # 실패한 회차들 재시도 (단일 스레드)
    if failed_rounds:
        status_text.text(f"실패한 {len(failed_rounds)}개 회차 재시도 중...")
        for round_num in failed_rounds:
            result = fetch_single_round(round_num)
            if result:
                data.append(result)
    
    # 회차 순서로 정렬
    data.sort(key=lambda x: x['round'])
    
    progress_bar.empty()
    status_text.empty()
    
    if failed_rounds:
        st.info(f"✅ 총 {len(data)}개 회차 수집 완료! (실패: {len(failed_rounds)}개)")
    else:
        st.success(f"✅ 총 {len(data)}개 회차 수집 완료!")
    
    return data

# ---------- 통계 분석 함수 ----------
def analyze_data(data):
    """수집된 데이터 분석"""
    if not data:
        return None, None, None
    
    # 전체 번호 수집
    all_numbers = []
    for entry in data:
        all_numbers.extend(entry['numbers'])
    
    # 빈도 분석
    frequency = Counter(all_numbers)
    
    # Overdue 분석 (각 번호가 마지막으로 나온 후 경과 회차)
    overdue = {}
    latest_round = max(entry['round'] for entry in data)
    
    for num in range(1, 46):
        last_appearance = 0
        for entry in reversed(data):  # 최신부터 검색
            if num in entry['numbers']:
                last_appearance = entry['round']
                break
        overdue[num] = latest_round - last_appearance if last_appearance > 0 else latest_round
    
    return frequency, overdue, all_numbers

# ---------- 7가지 알고리즘 ----------
def fibonacci_method(frequency, overdue):
    """피보나치 수열 기반 번호 선택"""
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34]
    
    # 피보나치 수와 일치하는 번호들 우선 선택
    fib_numbers = [num for num in range(1, 46) if num in fib_sequence]
    
    # 빈도와 overdue를 고려한 가중치 계산
    weighted_numbers = []
    for num in range(1, 46):
        weight = frequency.get(num, 0) * 0.3 + overdue.get(num, 0) * 0.7
        if num in fib_numbers:
            weight *= 1.5  # 피보나치 수에 가중치
        weighted_numbers.append((num, weight))
    
    # 가중치 기준 정렬 후 상위 6개 선택
    weighted_numbers.sort(key=lambda x: x[1], reverse=True)
    selected = [num for num, _ in weighted_numbers[:6]]
    
    return sorted(selected)

def einstein_entropy_blend(frequency, overdue):
    """아인슈타인 엔트로피 기반 번호 조합"""
    # E=mc²에서 영감을 받은 가중치 계산
    c_squared = 299792458 ** 2  # 광속의 제곱
    
    weighted_numbers = []
    for num in range(1, 46):
        # 질량(빈도)과 에너지(overdue)의 관계
        mass = frequency.get(num, 1)
        energy = overdue.get(num, 1)
        
        # 아인슈타인 공식 변형
        einstein_weight = (mass * energy) % 100  # 계산 결과를 적절한 범위로 조정
        weighted_numbers.append((num, einstein_weight))
    
    # 엔트로피 최대화를 위한 다양성 고려
    weighted_numbers.sort(key=lambda x: x[1], reverse=True)
    
    # 상위 12개 중에서 다양성을 고려하여 6개 선택
    top_candidates = [num for num, _ in weighted_numbers[:12]]
    selected = random.sample(top_candidates, 6)
    
    return sorted(selected)

def pythagoras_triangle_bias(frequency, overdue):
    """피타고라스 정리 기반 삼각수 편향"""
    # 삼각수 계산 (n*(n+1)/2)
    triangle_numbers = []
    n = 1
    while True:
        triangle = n * (n + 1) // 2
        if triangle > 45:
            break
        triangle_numbers.append(triangle)
        n += 1
    
    # 피타고라스 수 쌍 찾기
    pythagoras_numbers = []
    for a in range(1, 46):
        for b in range(a + 1, 46):
            c_squared = a*a + b*b
            c = int(c_squared ** 0.5)
            if c*c == c_squared and c <= 45:
                pythagoras_numbers.extend([a, b, c])
    
    # 가중치 계산
    weighted_numbers = []
    for num in range(1, 46):
        weight = frequency.get(num, 0) * 0.4 + overdue.get(num, 0) * 0.6
        
        # 삼각수나 피타고라스 수에 보너스
        if num in triangle_numbers:
            weight *= 1.3
        if num in pythagoras_numbers:
            weight *= 1.2
            
        weighted_numbers.append((num, weight))
    
    weighted_numbers.sort(key=lambda x: x[1], reverse=True)
    selected = [num for num, _ in weighted_numbers[:6]]
    
    return sorted(selected)

def nobel_diversity_optimizer(frequency, overdue):
    """노벨상 수상 연도 기반 다양성 최적화"""
    # 주요 노벨상 수상 연도의 마지막 두 자리
    nobel_years = [1, 3, 5, 8, 11, 15, 18, 21, 27, 29, 32, 35, 38, 41, 43, 45]
    
    # 구간별 다양성 확보 (1-10, 11-20, 21-30, 31-40, 41-45)
    ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]
    selected = []
    
    for start, end in ranges:
        range_numbers = []
        for num in range(start, min(end + 1, 46)):
            weight = frequency.get(num, 0) * 0.3 + overdue.get(num, 0) * 0.7
            if num in nobel_years:
                weight *= 1.4
            range_numbers.append((num, weight))
        
        if range_numbers:
            range_numbers.sort(key=lambda x: x[1], reverse=True)
            # 각 구간에서 1개씩 선택 (마지막 구간은 2개)
            count = 2 if start == 41 else 1
            for i in range(min(count, len(range_numbers))):
                if len(selected) < 6:
                    selected.append(range_numbers[i][0])
    
    # 6개가 안 되면 추가 선택
    if len(selected) < 6:
        remaining = [num for num in range(1, 46) if num not in selected]
        additional_needed = 6 - len(selected)
        additional = random.sample(remaining, min(additional_needed, len(remaining)))
        selected.extend(additional)
    
    return sorted(selected[:6])

def monte_carlo_simulation(frequency, overdue):
    """몬테카를로 시뮬레이션"""
    # 여러 번의 시뮬레이션을 통한 확률적 선택
    simulation_results = defaultdict(int)
    num_simulations = 1000
    
    for _ in range(num_simulations):
        # 각 번호의 선택 확률 계산
        probabilities = []
        total_weight = 0
        
        for num in range(1, 46):
            weight = frequency.get(num, 1) * 0.4 + overdue.get(num, 1) * 0.6
            probabilities.append(weight)
            total_weight += weight
        
        # 확률 정규화
        probabilities = [p / total_weight for p in probabilities]
        
        # 가중 확률을 사용한 6개 번호 선택
        selected_numbers = []
        available_numbers = list(range(1, 46))
        available_probs = probabilities[:]
        
        for _ in range(6):
            # 확률에 따른 선택
            choice_idx = random.choices(range(len(available_numbers)), 
                                       weights=available_probs, k=1)[0]
            selected_num = available_numbers[choice_idx]
            selected_numbers.append(selected_num)
            
            # 선택된 번호 제거
            available_numbers.pop(choice_idx)
            available_probs.pop(choice_idx)
        
        # 결과 집계
        for num in selected_numbers:
            simulation_results[num] += 1
    
    # 가장 많이 선택된 6개 번호
    top_numbers = sorted(simulation_results.items(), key=lambda x: x[1], reverse=True)[:6]
    selected = [num for num, count in top_numbers]
    
    return sorted(selected)

def edward_thorp_strategy(frequency, overdue):
    """에드워드 소프의 카드 카운팅 이론 적용"""
    # 켈리 공식을 로또에 적용한 버전
    
    # 각 번호의 "카운트"를 계산 (빈도 기반)
    total_draws = sum(frequency.values()) // 6  # 총 추첨 횟수
    
    # 기댓값 계산
    expected_freq = total_draws / 45 * 6  # 각 번호의 기대 출현 횟수
    
    # 켈리 기준 적용: (bp - q) / b
    # b = 배당률(단순화하여 1 사용), p = 성공확률, q = 실패확률
    kelly_scores = []
    
    for num in range(1, 46):
        actual_freq = frequency.get(num, 0)
        overdue_score = overdue.get(num, 0)
        
        # 성공 확률 추정 (빈도와 overdue를 종합)
        p = (actual_freq / expected_freq) * 0.6 + (overdue_score / max(overdue.values())) * 0.4
        p = max(0.01, min(0.99, p))  # 확률 범위 제한
        
        q = 1 - p
        kelly_score = (p - q)  # 단순화된 켈리 공식
        
        kelly_scores.append((num, kelly_score))
    
    # 켈리 점수 기준 정렬
    kelly_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 상위 8개 중에서 6개를 랜덤 선택 (리스크 분산)
    top_candidates = [num for num, _ in kelly_scores[:8]]
    selected = random.sample(top_candidates, 6)
    
    return sorted(selected)

def blaise_pascal_probability(frequency, overdue):
    """블레즈 파스칼의 확률론과 조합론 적용"""
    
    # 파스칼 삼각형의 조합 계수 활용
    def combination(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        
        result = 1
        for i in range(min(r, n - r)):
            result = result * (n - i) // (i + 1)
        return result
    
    # 각 번호에 조합론적 가중치 부여
    weighted_numbers = []
    
    for num in range(1, 46):
        # 기본 통계적 가중치
        base_weight = frequency.get(num, 0) * 0.5 + overdue.get(num, 0) * 0.5
        
        # 조합론적 가중치 (파스칼 삼각형에서의 위치)
        # C(45, num)을 단순화하여 적용
        combination_weight = combination(45, num % 10) / 1000  # 정규화
        
        # 확률론적 조정
        # 번호를 구간별로 나누어 균형 잡힌 선택 유도
        zone = (num - 1) // 9  # 0~4 구간
        zone_balance = 1 + (zone * 0.1)  # 구간별 균형 조정
        
        total_weight = base_weight + combination_weight * zone_balance
        weighted_numbers.append((num, total_weight))
    
    # 파스칼의 확률 원리: 기댓값 최대화
    weighted_numbers.sort(key=lambda x: x[1], reverse=True)
    
    # 조합론적 다양성 고려: 연속된 번호 최소화
    selected = []
    candidates = [num for num, _ in weighted_numbers]
    
    # 첫 번째 번호 선택
    selected.append(candidates[0])
    
    # 나머지 5개 선택 시 연속성 고려
    for candidate in candidates[1:]:
        if len(selected) >= 6:
            break
            
        # 이미 선택된 번호와 연속되지 않는지 확인
        is_consecutive = any(abs(candidate - sel) == 1 for sel in selected)
        
        if not is_consecutive or len(selected) >= 5:  # 마지막 번호는 연속성 무시
            selected.append(candidate)
    
    # 6개가 안 되면 나머지 추가
    if len(selected) < 6:
        remaining = [num for num in range(1, 46) if num not in selected]
        needed = 6 - len(selected)
        selected.extend(random.sample(remaining, min(needed, len(remaining))))
    
    return sorted(selected[:6])

# ---------- Streamlit 메인 앱 ----------
def main():
    # 타이틀과 설명
    st.title("🎯 로또 분석 추천 시스템 v2.0")
    st.markdown("### 7가지 수학적 알고리즘으로 로또 번호를 분석하고 추천합니다")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 설정")
    
    # 최신 회차 정보
    latest_round = get_latest_round()
    st.sidebar.info(f"📅 최신 회차: {latest_round}회차")
    
    # 회차 범위 설정
    st.sidebar.subheader("📊 데이터 수집 범위")
    
    # 빠른 선택 옵션 추가
    quick_options = st.sidebar.selectbox(
        "🚀 빠른 선택", 
        ["사용자 정의", "최근 100회차 (빠름)", "최근 200회차 (보통)", "최근 500회차 (느림)", "전체 데이터 (매우 느림)"]
    )
    
    if quick_options == "최근 100회차 (빠름)":
        start_round = max(1, latest_round - 99)
        end_round = latest_round
    elif quick_options == "최근 200회차 (보통)":
        start_round = max(1, latest_round - 199)
        end_round = latest_round
    elif quick_options == "최근 500회차 (느림)":
        start_round = max(1, latest_round - 499)
        end_round = latest_round
    elif quick_options == "전체 데이터 (매우 느림)":
        start_round = 1
        end_round = latest_round
    else:  # 사용자 정의
        start_round = st.sidebar.number_input("시작 회차", min_value=1, max_value=latest_round, value=max(1, latest_round-99))
        end_round = st.sidebar.number_input("종료 회차", min_value=start_round, max_value=latest_round, value=latest_round)
    
    # 예상 소요 시간 표시
    total_rounds = end_round - start_round + 1
    estimated_time = total_rounds * 0.2  # 병렬 처리로 회차당 0.2초 예상
    if estimated_time < 60:
        time_str = f"약 {estimated_time:.0f}초"
    else:
        time_str = f"약 {estimated_time/60:.1f}분"
    
    st.sidebar.info(f"📊 수집 회차: {total_rounds}개\n⏱️ 예상 시간: {time_str}")
    
    if total_rounds > 500:
        st.sidebar.warning("⚠️ 500회차 이상은 시간이 오래 걸립니다!")
    
    # 데이터 수집 버튼
    if st.sidebar.button("🔄 데이터 수집", type="primary"):
        st.session_state.lotto_data = None  # 캐시 초기화
        
        with st.spinner("데이터를 수집하고 있습니다..."):
            data = fetch_lotto_data(start_round, end_round)
            
        if data:
            st.session_state.lotto_data = data
            st.session_state.frequency, st.session_state.overdue, st.session_state.all_numbers = analyze_data(data)
            
            # 자동으로 모든 알고리즘 번호 생성
            algorithms = {
                "🌀 Fibonacci Method": fibonacci_method,
                "🧠 Einstein Entropy Blend": einstein_entropy_blend,
                "📐 Pythagoras Triangle Bias": pythagoras_triangle_bias,
                "🏆 Nobel Diversity Optimizer": nobel_diversity_optimizer,
                "🎲 Monte Carlo Simulation": monte_carlo_simulation,
                "💰 Edward Thorp Strategy": edward_thorp_strategy,
                "🎯 Blaise Pascal Probability": blaise_pascal_probability
            }
            
            # 모든 알고리즘 실행하여 추천 번호 자동 생성
            for name, algo_func in algorithms.items():
                numbers = algo_func(st.session_state.frequency, st.session_state.overdue)
                st.session_state[f"numbers_{name}"] = numbers
            
            st.sidebar.success(f"✅ {len(data)}개 회차 데이터 수집 완료!")
            st.sidebar.success("🎯 모든 알고리즘 번호 자동 생성 완료!")
        else:
            st.sidebar.error("❌ 데이터 수집에 실패했습니다.")
    
    # 메인 콘텐츠
    if 'lotto_data' in st.session_state and st.session_state.lotto_data:
        # 탭 생성
        tab1, tab2, tab3, tab4 = st.tabs(["📊 통계 분석", "🎯 번호 추천", "📈 시각화", "💾 데이터 다운로드"])
        
        with tab1:
            st.header("📊 통계 분석 결과")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔢 빈도 분석 (Top 10)")
                freq_df = pd.DataFrame(list(st.session_state.frequency.items()), columns=['번호', '출현횟수'])
                freq_df = freq_df.sort_values('출현횟수', ascending=False).head(10)
                st.dataframe(freq_df, use_container_width=True)
            
            with col2:
                st.subheader("⏰ Overdue 분석 (Top 10)")
                overdue_df = pd.DataFrame(list(st.session_state.overdue.items()), columns=['번호', '미출현회차'])
                overdue_df = overdue_df.sort_values('미출현회차', ascending=False).head(10)
                st.dataframe(overdue_df, use_container_width=True)
        
        with tab2:
            st.header("🎯 7가지 알고리즘 번호 추천")
            
            algorithms = {
                "🌀 Fibonacci Method": fibonacci_method,
                "🧠 Einstein Entropy Blend": einstein_entropy_blend,
                "📐 Pythagoras Triangle Bias": pythagoras_triangle_bias,
                "🏆 Nobel Diversity Optimizer": nobel_diversity_optimizer,
                "🎲 Monte Carlo Simulation": monte_carlo_simulation,
                "💰 Edward Thorp Strategy": edward_thorp_strategy,
                "🎯 Blaise Pascal Probability": blaise_pascal_probability
            }
            
            # 데이터가 있으면 알고리즘별 추천 번호 표시
            if 'frequency' in st.session_state and 'overdue' in st.session_state:
                # 모든 알고리즘 재생성 버튼 (상단에 배치)
                col_top1, col_top2 = st.columns([3, 1])
                with col_top2:
                    if st.button("🔄 모든 번호 재생성", type="secondary"):
                        for name, algo_func in algorithms.items():
                            numbers = algo_func(st.session_state.frequency, st.session_state.overdue)
                            st.session_state[f"numbers_{name}"] = numbers
                        st.rerun()
                
                st.divider()
                
                # 알고리즘별 추천 번호 표시
                for name, algo_func in algorithms.items():
                    with st.expander(name, expanded=True):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            # 자동 생성된 번호가 있으면 표시, 없으면 생성
                            if f"numbers_{name}" not in st.session_state:
                                numbers = algo_func(st.session_state.frequency, st.session_state.overdue)
                                st.session_state[f"numbers_{name}"] = numbers
                            
                            numbers = st.session_state[f"numbers_{name}"]
                            # 번호를 예쁘게 표시
                            number_html = " ".join([f'<span style="background-color: #ff6b6b; color: white; padding: 5px 10px; border-radius: 20px; margin: 2px; display: inline-block; font-weight: bold;">{num}</span>' for num in numbers])
                            st.markdown(number_html, unsafe_allow_html=True)
                        
                        with col2:
                            if st.button(f"재생성", key=f"regen_{name}"):
                                numbers = algo_func(st.session_state.frequency, st.session_state.overdue)
                                st.session_state[f"numbers_{name}"] = numbers
                                st.rerun()
            else:
                st.info("👈 왼쪽 사이드바에서 데이터 수집을 먼저 진행해주세요!")
        
        with tab3:
            st.header("📈 데이터 시각화")
            
            # 빈도 차트
            fig_freq = px.bar(freq_df, x='번호', y='출현횟수', 
                             title="번호별 출현 빈도 (Top 10)")
            st.plotly_chart(fig_freq, use_container_width=True)
            
            # Overdue 차트
            fig_overdue = px.bar(overdue_df, x='번호', y='미출현회차',
                               title="번호별 미출현 회차 (Top 10)")
            st.plotly_chart(fig_overdue, use_container_width=True)
        
        with tab4:
            st.header("💾 데이터 다운로드")
            
            # 전체 분석 데이터 CSV 생성
            all_data = []
            for num in range(1, 46):
                all_data.append({
                    '번호': num,
                    '출현횟수': st.session_state.frequency.get(num, 0),
                    '미출현회차': st.session_state.overdue.get(num, 0)
                })
            
            df_download = pd.DataFrame(all_data)
            
            # CSV 생성 시 UTF-8 BOM 추가로 한글 깨짐 방지
            output = io.StringIO()
            df_download.to_csv(output, index=False, encoding='utf-8')
            csv_string = output.getvalue()
            
            # UTF-8 BOM 추가
            csv = '\ufeff' + csv_string
            
            st.download_button(
                label="📊 분석 데이터 다운로드 (CSV)",
                data=csv.encode('utf-8'),
                file_name=f"lotto_analysis_{start_round}_{end_round}.csv",
                mime="text/csv"
            )
            
            # 추천 번호 다운로드
            if any(f"numbers_{name}" in st.session_state for name in algorithms.keys()):
                recommendation_data = []
                for name in algorithms.keys():
                    if f"numbers_{name}" in st.session_state:
                        numbers = st.session_state[f"numbers_{name}"]
                        recommendation_data.append({
                            '알고리즘': name,
                            '추천번호': ', '.join(map(str, numbers))
                        })
                
                if recommendation_data:
                    rec_df = pd.DataFrame(recommendation_data)
                    
                    # CSV 생성 시 UTF-8 BOM 추가로 한글 깨짐 방지
                    output = io.StringIO()
                    rec_df.to_csv(output, index=False, encoding='utf-8')
                    csv_string = output.getvalue()
                    
                    # UTF-8 BOM 추가
                    rec_csv = '\ufeff' + csv_string
                    
                    st.download_button(
                        label="🎯 추천 번호 다운로드 (CSV)",
                        data=rec_csv.encode('utf-8'),
                        file_name=f"lotto_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    else:
        # 초기 화면
        st.success("🎯 **자동 번호 생성 시스템** - 데이터 수집 후 모든 알고리즘이 자동으로 실행됩니다!")
        st.info("👈 왼쪽 사이드바에서 **'🔄 데이터 수집'** 버튼을 클릭하세요!")
        
        # 사용 순서 안내
        st.markdown("""
        ## 📋 사용 순서
        1. **왼쪽 사이드바**에서 데이터 수집 범위 선택
        2. **'🔄 데이터 수집'** 버튼 클릭
        3. **자동으로 7가지 알고리즘 번호 생성**
        4. **'🎯 번호 추천'** 탭에서 모든 결과 확인
        """)
        
        # 기능 소개
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🔢 7가지 알고리즘
            - Fibonacci Method
            - Einstein Entropy Blend  
            - Pythagoras Triangle Bias
            - Nobel Diversity Optimizer
            """)
        
        with col2:
            st.markdown("""
            ### 🔢 추가 알고리즘
            - Monte Carlo Simulation
            - Edward Thorp Strategy
            - Blaise Pascal Probability
            """)
        
        with col3:
            st.markdown("""
            ### 📊 분석 기능
            - 🚀 **자동 번호 생성**
            - 실시간 데이터 수집
            - 빈도/Overdue 분석
            - 시각화 차트
            - CSV 데이터 다운로드
            """)
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🎯 로또 분석 추천 시스템 v2.0 | 
        ⚠️ 본 프로그램은 통계적 분석 도구이며 당첨을 보장하지 않습니다 |
        📅 2025.10.23
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()
