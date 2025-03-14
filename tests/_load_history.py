import shelve
import os
import stat
import dbm

memory_path = "/home/conan/workspace/dcai-mvp-server/qdrant_storage"
memory_key = "2ce13ce1-feff-4456-be52-793969f2bd09"

# 파일의 읽기 권한을 추가
# os.chmod(f'{memory_path}/{memory_key}', stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
# print(dbm.whichdb(f'{memory_path}/{memory_key}'))
# dbm.open(f'{memory_path}/{memory_key}', 'c').close()
# with shelve.open(f'{memory_path}/{memory_key}', flag='r') as db:
#     print(db['history'])

with open(f'{memory_path}/{memory_key}', 'r', encoding='utf-8', errors='ignore') as f:
    data = f.read()
    print(type(data))
    print(data)
