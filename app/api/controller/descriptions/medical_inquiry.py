chat_description = '''
# Medical Inquiry Chat API

## Request Parameters
- **uid** (string, required)
  - 사용자 식별자
  
- **state** (integer, required)
  - 채팅 상태값
  - 상태 코드:
    - 0: 시작
    - 1: 증상 부위 입력
    - 2: 설문 진행 중
    - 3: 요약
    - 4: 치료 방법
    - 5: 종료

- **text** (string, required)
  - 사용자 입력 텍스트

- **lang** (string, optional)
  - 응답 언어 설정
  - 기본값: "ko"
  - 허용값:
    - "ko": 한국어
    - "en": 영어

## Response Format
- **text** (string)
  - 챗봇 응답 텍스트

- **screening** (array[object], optional)
  - 문진 결과
  - 객체 구조:
    ```json
    {
      "label": string,
      "content": string | null
    }
    ```
  - 포함 항목:
    - 증상
    - 증상 강도
    - 증상 부위
    - 지속 기간
    - 증상 유발요인
    - 하고 싶은 말

- **treatment** (array[string], optional)
  - 추천 치료 방법 목록

- **state** (integer)
  - 현재 대화 상태값
  - 다음 요청시 이 값을 사용

## Usage Notes
1. 첫 요청시 state=0으로 시작
2. 이후 요청은 이전 응답의 state 값을 사용하여 채팅 진행
3. screening 결과는 문진 단계에서만 반환
4. treatment 정보는 치료 방법 추천 단계에서만 반환
'''