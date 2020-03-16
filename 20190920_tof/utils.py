#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
# logger.setLevel(level=logging.INFO)

# --- 创建一个FileHandler，并对输出消息的格式进行设置，将其添加到logger，然后将日志写入到指定的文件中
# file_handler = logging.FileHandler("log.txt")
# file_handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# --- logger中添加StreamHandler，可以将日志输出到屏幕上
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
