
import logging

_level = logging.INFO
# _level = logging.DEBUG

class CustomFilter(logging.Filter):
    def filter(self, record):
        # 忽略包含 "HTTP/1.1 200 OK" 的日志消息
        if "HTTP/1.1 200 OK" in record.getMessage():
            return False
        return True

logging.basicConfig(level=_level, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLMChat")

logger2 = logging.getLogger("httpx")
logger2.addFilter(CustomFilter())