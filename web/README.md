# 🚒 소방 인트라넷 랜딩페이지

Whisper AI 기반 119 긴급전화 음성인식 시스템을 위한 랜딩페이지입니다.

## ✨ 주요 기능

- **배경 동영상 재생**: 전체 화면 배경에 동영상이 자동으로 재생됩니다
- **동영상 제어**: 우측 하단 버튼으로 동영상 재생/일시정지 가능
- **반응형 디자인**: 모바일, 태블릿, 데스크톱 모든 기기 지원
- **부드러운 애니메이션**: 스크롤 시 요소들이 자연스럽게 나타남
- **현대적인 UI**: 그라디언트 오버레이와 글래스모피즘 효과

## 📁 파일 구조

```
web/
├── index.html       # 메인 HTML 파일
├── styles.css       # CSS 스타일시트
├── script.js        # JavaScript 기능
├── background.mp4   # 배경 동영상 (직접 추가 필요)
└── README.md        # 이 파일
```

## 🚀 사용 방법

### 1. 배경 동영상 추가

`web/` 디렉토리에 `background.mp4` 파일을 추가해주세요.

#### 추천 동영상 소스:
- **Pexels**: https://www.pexels.com/ko-kr/search/videos/소방/
- **Pixabay**: https://pixabay.com/ko/videos/
- **Videvo**: https://www.videvo.net/

#### 추천 키워드:
- 소방관 (firefighter)
- 긴급 상황 (emergency)
- 소방차 (fire truck)
- 119 응급 상황
- 도시 야경 (city night)

#### 동영상 사양 권장사항:
- 해상도: 1920x1080 (Full HD) 이상
- 길이: 10-30초 (루프 재생)
- 포맷: MP4 (H.264 코덱)
- 파일 크기: 10MB 이하 (로딩 속도 최적화)

### 2. 웹 서버 실행

#### Python 내장 웹 서버 사용:

```bash
cd web
python -m http.server 8000
```

#### Node.js http-server 사용:

```bash
cd web
npx http-server -p 8000
```

### 3. 브라우저에서 접속

브라우저를 열고 다음 주소로 접속:

```
http://localhost:8000
```

## 🎨 커스터마이징

### 색상 변경

`styles.css` 파일에서 다음 색상을 변경할 수 있습니다:

```css
/* 주요 색상 (빨간색 계열 - 소방 테마) */
#dc2626  /* 진한 빨강 */
#ef4444  /* 밝은 빨강 */

/* 배경 색상 */
#0f172a  /* 진한 네이비 */
#1e293b  /* 네이비 */
```

### 텍스트 수정

`index.html` 파일에서 다음 내용을 수정할 수 있습니다:
- 제목 및 부제목
- 기능 설명
- 네비게이션 메뉴
- 버튼 텍스트

### 동영상 없이 실행

동영상 파일이 없어도 페이지는 정상적으로 작동합니다.
동영상 대신 그라디언트 배경이 자동으로 표시됩니다.

## 🔧 기술 스택

- **HTML5**: 시맨틱 마크업
- **CSS3**:
  - Flexbox & Grid 레이아웃
  - CSS 애니메이션
  - Backdrop Filter (글래스모피즘)
  - 반응형 미디어 쿼리
- **JavaScript (ES6+)**:
  - DOM 조작
  - Intersection Observer API
  - 이벤트 핸들링

## 📱 브라우저 호환성

- Chrome/Edge: ✅ 완벽 지원
- Firefox: ✅ 완벽 지원
- Safari: ✅ 완벽 지원
- 모바일 브라우저: ✅ 완벽 지원

## 🔐 보안 고려사항

실제 운영 환경에서는 다음 사항을 고려해주세요:

1. HTTPS 사용 (SSL/TLS 인증서)
2. CSP (Content Security Policy) 헤더 설정
3. XSS 방지를 위한 입력 검증
4. 적절한 CORS 설정

## 🎯 향후 개발 계획

- [ ] Whisper API 연동
- [ ] 실시간 음성 인식 데모
- [ ] 관리자 대시보드
- [ ] 통계 및 분석 페이지
- [ ] 다크/라이트 모드 토글

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

**Powered by OpenAI Whisper** | © 2026 소방청
