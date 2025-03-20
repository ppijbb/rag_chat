import logging
import sys
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def get_logger():
    """로깅 설정을 구성합니다."""
    logger = logging.getLogger("my_app")
    
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 파일 핸들러 (선택 사항)
    # file_handler = logging.FileHandler("app.log")
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    return logger
