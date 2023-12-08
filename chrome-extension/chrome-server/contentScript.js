window.onload = function () {
    // 뉴스 페이지에서 제목과 내용 수집 (일단 테스트를 위해 임시 제목, 내용 전달)
    const title = '4000만원대 \'갓성비\' 입소문에…인기 폭발한 독일'
    const content = '차량 온도 조절은 \'3존 클리마트로닉 자동 에어컨\'으로 좌석마다 각기 조절할 수 있다. 앞좌석 통풍 시트와 앞·뒷좌석 히팅 시트가 적용됐다. 여기에 테일게이트를 손쉽게 여닫을 수 있는 \'트렁크 이지 오픈 앤 클로즈\'가 기본 제공된다.파워트레인은 도심·험로를 넘나드는 파워와 정숙성, 효율을 고루 갖춘 \'2.0 TSI 가솔린\' 엔진이 탑재됐다. 특히 2023년형 티구안 올스페이스 2.0 TSI 프레스티지 모델은 8단 자동변속기와 결합해 최고 출력 186마력(4400~6000rpm)과 최대토크 30.6㎏.m의 여유로운 힘과 안정적인 주행 질감을 제공한다.티구안 올스페이스 가솔린은 중형 SUV 급 차체에 강력한 엔진을 탑재하고도 리터당 복합 10.1㎞(도심 9.0㎞/ℓ, 고속 11.9㎞/ℓ)의 연비를 인증받았다. 가솔린 모델은 저공해 3종 친환경 차로 분류돼 공영주차장, 서울 지하철 환승주차장, 공항주차장 할인을 받을 수 있다.'
    // 데이터 전송
    sendDataToServer(title, content);
  };
  
  // 서버로 데이터 전송
  function sendDataToServer(title, content) {
    const serverUrl = 'http://localhost:8000/scrape';
  
    fetch(serverUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ title, content }),
      mode: 'cors',
    })
      .then(response => response.json())
      .then(data => {
        // 서버로부터 받은 결과 처리
        console.log('Data received from the server:', data);
      })
      .catch(error => {
        console.error('Error sending data to the server:', error);
      });
  }

  