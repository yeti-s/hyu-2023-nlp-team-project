### Chrome Extension

#### 1. chrome-extension
`main.py` - fastAPI 서버

[실행 방법]<br>
`uvicorn main:app --reload`

[동작 방식]<br>
1. Hugging Face Hub로부터 pipeline을 받아옵니다.

```python
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification",model="yeti-s/clickbait_detector")
```
2. 클라이언트로부터 받은 내용을 위 모듈에 전달합니다.
3. 모듈로부터 받은 결과를 다시 클라이언트에 전달합니다.

<br>

#### 2. chrome-extension/chrome-server 폴더
1. `manifest.json` : 크롬 익스텐션 기본 설정 파일
2. `contentScript.js` : 특정 url 진입 시 실행되는 스크립트