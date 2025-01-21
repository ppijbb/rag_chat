import datetime

days_kor = ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")
def format_datetime_with_ampm(dt):
    """
    datetime 객체를 받아서 오전/오후가 포함된 문자열로 반환
    """
    hour = dt.hour
    if hour > 12:
        hour -= 12
    ampm = "오전" if dt.hour < 12 else "오후"
    return f"{dt.year}년 {dt.month}월 {dt.day}일 {days_kor[dt.weekday()]} {ampm} {hour}시 {dt.minute}분"
