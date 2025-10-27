# � 로또 6/45 번호 추천 시스템

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lotto-recommendation-system.streamlit.app/)

## � 프로젝트 소개

동행복권 API를 활용한 로또 6/45 번호 분석 및 추천 시스템입니다. 7가지 수학적 알고리즘을 통해 과학적으로 번호를 추천합니다.

## ✨ 주요 기능

### 🔢 7가지 알고리즘
- **Fibonacci Method**: 피보나치 수열 기반 분석
- **Einstein Entropy Blend**: 엔트로피 이론 활용
- **Pythagoras Triangle Bias**: 피타고라스 삼각형 편향
- **Nobel Diversity Optimizer**: 다양성 최적화
- **Monte Carlo Simulation**: 몬테카를로 시뮬레이션
- **Edward Thorp Strategy**: 켈리 공식 기반
- **Blaise Pascal Probability**: 조합론적 확률

### 📊 분석 기능
- 🚀 **자동 번호 생성**: 데이터 수집 후 모든 알고리즘 자동 실행
- 📈 **실시간 데이터**: 동행복권 API를 통한 최신 당첨 번호 수집
- 📊 **시각화**: 번호별 출현 빈도 및 미출현 회차 차트
- 💾 **데이터 다운로드**: CSV 형태로 분석 결과 및 추천 번호 다운로드

### 🎨 사용자 경험
- ⚡ **고성능**: 병렬 처리로 데이터 수집 시간 25배 단축 (2시간 → 4-5분)
- 🌐 **웹 기반**: 브라우저에서 바로 사용 가능
- 📱 **반응형**: 모바일 및 데스크톱 지원
- 🎯 **직관적 UI**: 탭 기반 인터페이스로 쉬운 사용

## 🚀 사용 방법

### 온라인 사용 (추천)
1. [웹사이트 접속](https://lotto-recommendation-system.streamlit.app/)
2. 왼쪽 사이드바에서 데이터 수집 범위 선택
3. '🔄 데이터 수집' 버튼 클릭
4. 자동으로 생성된 추천 번호 확인

### 로컬 실행
```bash
# 저장소 클론
git clone https://github.com/username/lotto-recommendation-system.git
cd lotto-recommendation-system

# 의존성 설치
pip install -r requirements.txt

# 앱 실행
streamlit run app.py
```

## 📁 프로젝트 구조

```
├── app.py                 # 메인 Streamlit 애플리케이션
├── requirements.txt       # Python 의존성
├── .streamlit/
│   └── config.toml       # Streamlit 설정
├── favicon.ico           # 웹사이트 아이콘
└── README.md            # 프로젝트 문서
```

## 🛠 기술 스택

- **Frontend**: Streamlit
- **Backend**: Python
- **Data Processing**: Pandas, Plotly
- **API**: 동행복권 공식 API
- **Deployment**: Streamlit Community Cloud

## 📊 성능 최적화

- **병렬 처리**: ThreadPoolExecutor를 활용한 API 호출 최적화
- **메모리 효율**: 세션 상태 관리로 중복 계산 방지
- **사용자 경험**: 자동 번호 생성으로 클릭 횟수 최소화

## 🔒 데이터 정책

- 동행복권 공식 API만 사용
- 개인정보 수집 없음
- 모든 데이터는 공개 당첨 정보만 활용

## ⚠️ 면책 조항

이 시스템은 교육 및 연구 목적으로 제작되었습니다. 로또는 확률 게임이므로 당첨을 보장하지 않습니다. 책임감 있는 게임 참여를 권장합니다.

## 📞 문의

프로젝트에 대한 문의사항이나 개선 제안이 있으시면 이슈를 생성해 주세요.

---

**⭐ 이 프로젝트가 도움이 되셨다면 Star를 눌러주세요!**
1. 링크 클릭
2. 사이드바에서 데이터 수집
3. 원하는 알고리즘으로 번호 추천
4. 결과 다운로드 가능

⚠️ 참고용 통계 도구 (당첨 보장 안됨)
```

## 🆚 exe vs 웹앱 비교

| 항목 | Windows exe | Streamlit 웹앱 |
|------|-------------|----------------|
| **설치** | 다운로드 필요 | 브라우저만 있으면 OK |
| **공유** | 파일 전송 어려움 | URL 링크로 간단 |
| **접근성** | Windows만 | 모든 OS/기기 |
| **업데이트** | 재배포 필요 | 자동 업데이트 |
| **보안** | Windows 경고 | 브라우저 기반 안전 |
| **성능** | 빠름 | 네트워크 의존 |

## 🎯 최종 권장사항

**웹앱 배포 시 Streamlit Cloud 사용을 강력 추천!**
- ✅ 완전 무료
- ✅ GitHub 연동 간단
- ✅ 자동 HTTPS
- ✅ 안정적인 호스팅
- ✅ 커뮤니티 지원

---

**🚀 이제 exe 파일 걱정 없이 URL 하나로 누구나 접속할 수 있습니다!**