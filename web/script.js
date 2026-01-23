// DOM 로딩 완료 후 실행
document.addEventListener('DOMContentLoaded', function() {
    // 동영상 요소 가져오기
    const video = document.getElementById('bgVideo');
    const videoToggle = document.getElementById('videoToggle');
    const playIcon = document.getElementById('playIcon');

    // 동영상 재생/일시정지 토글
    if (videoToggle && video) {
        videoToggle.addEventListener('click', function() {
            if (video.paused) {
                video.play();
                playIcon.textContent = '⏸';
                videoToggle.title = '동영상 일시정지';
            } else {
                video.pause();
                playIcon.textContent = '▶';
                videoToggle.title = '동영상 재생';
            }
        });
    }

    // 동영상 로드 실패 시 처리
    if (video) {
        video.addEventListener('error', function() {
            console.log('동영상 로드 실패 - 배경 그라디언트로 대체');
            const videoBackground = document.querySelector('.video-background');
            if (videoBackground) {
                videoBackground.style.background = 'linear-gradient(135deg, #1e3a8a 0%, #dc2626 50%, #1e293b 100%)';
            }
        });

        // 동영상이 성공적으로 로드되었을 때
        video.addEventListener('loadeddata', function() {
            console.log('배경 동영상 로드 완료');
        });

        // 동영상 자동 재생 시도
        video.play().catch(function(error) {
            console.log('자동 재생 실패:', error);
            // 일부 브라우저에서는 사용자 상호작용 후에만 재생 가능
        });
    }

    // 네비게이션 메뉴 스크롤 효과
    const navLinks = document.querySelectorAll('.nav-menu a');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href.startsWith('#')) {
                e.preventDefault();
                const targetId = href.substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });

    // 버튼 클릭 효과
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            // 버튼 클릭 시 실행할 동작
            const buttonText = this.textContent.trim();

            if (buttonText === '시스템 시작') {
                alert('119 음성인식 시스템을 시작합니다.\n\n(실제 시스템 연동은 별도 구현이 필요합니다)');
            } else if (buttonText === '자세히 보기') {
                const aboutSection = document.getElementById('about');
                if (aboutSection) {
                    aboutSection.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });

    // 스크롤 시 헤더 스타일 변경
    let lastScrollTop = 0;
    const header = document.querySelector('header');

    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

        if (scrollTop > 100) {
            header.style.background = 'rgba(15, 23, 42, 0.9)';
            header.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.3)';
        } else {
            header.style.background = 'rgba(15, 23, 42, 0.6)';
            header.style.boxShadow = 'none';
        }

        lastScrollTop = scrollTop;
    }, false);

    // 기능 카드 애니메이션 (스크롤 시)
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animation = 'fadeInUp 0.6s ease-out forwards';
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach(card => {
        card.style.opacity = '0';
        observer.observe(card);
    });

    // 페이지 로드 완료 메시지
    console.log('🚒 소방 인트라넷 랜딩페이지 로드 완료');
    console.log('📹 배경 동영상: background.mp4 (파일을 추가해주세요)');
});
