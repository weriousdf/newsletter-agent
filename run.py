import os
import sys

from graph import run                 # graph.py 에 있는 그 run()

if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        os.environ["DRY_RUN"] = "1"    # 35장의 dry_run 스위치
    out = run()
    for line in out["log"]:
        print(line)                    # 이 출력이 Actions 로그에 그대로 남는다
