

from loguru import logger


logger.remove()
logger.add("test.log")
a = logger._core.handlers
print(a)


for handler_id, handler in logger._core.handlers.items():
    print(handler._name)