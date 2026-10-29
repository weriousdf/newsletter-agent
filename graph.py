"""매일 아침 AI 소식을 골라 Discord로 보내는 뉴스레터 에이전트.

collect(수집) → select(선별) → report(취재·요약) → verify(검수) → publish(발행)
D00 교안 1~6부의 최종 코드를 한 파일로 옮겼다. 실행은 run.py 에서만 한다.
"""
import operator
import os
import re
import json
from datetime import datetime, timedelta, timezone
from typing import Annotated, TypedDict

import feedparser
import requests
import trafilatura
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

load_dotenv()                                   # .env 의 OPENAI_API_KEY 등을 읽는다
client = OpenAI()                               # 키는 OPENAI_API_KEY 환경변수에서 읽는다
UA = {"User-Agent": "Mozilla/5.0 (newsletter-agent-course)"}

# ── State ────────────────────────────────────────────────────────────
class Brief(TypedDict):
    hours:     int                              # 수집 time window(시간)
    collected: list                             # ① 수집한 기사
    picked:    list                             # ② 선별해 남긴 다섯 건
    drafted:   Annotated[list, operator.add]    # ③ 취재한 초안 — 워커들이 나눠 채운다
    verified:  list                             # ④ 검수를 통과한 것
    log:       Annotated[list, operator.add]    # 무슨 일이 있었는지

INIT = {"hours": 24, "collected": [], "picked": [], "drafted": [],
        "verified": [], "log": []}

# ── ① collect ────────────────────────────────────────────────────────
SOURCES = [
    ("Seen and Heard",     "https://seenandheard-international.com/feed/"),
    ("NY Classical Review", "https://newyorkclassicalreview.com/feed/"),
    ("SF Classical Voice", "https://www.sfcv.org/rss.xml"),
    ("Guardian Classical", "https://www.theguardian.com/music/classicalmusicandopera/rss"),
    ("I Care If You Listen", "https://icareifyoulisten.com/feed/"),
    ("Pizzicato",          "https://www.pizzicato.lu/feed/"),
    ("Slipped Disc",       "https://slippedisc.com/feed/"),
    ("연합뉴스 문화",         "https://www.yna.co.kr/rss/culture.xml"),
]

# 문화 전반을 다루는 소스는 클래식 키워드가 제목·요약에 있는 글만 남긴다
KEYWORD_FILTER = {
    "연합뉴스 문화": ["피아니스트", "오케스트라", "협연", "지휘자", "교향악", "필하모닉",
                  "리사이틀", "첼리스트", "바이올리니스트", "성악가", "임윤찬", "조성진"],
}

def strip_tags(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()

def published_at(entry):
    t = entry.get("published_parsed")
    return datetime(*t[:6], tzinfo=timezone.utc) if t else None

def collect(s: dict) -> dict:                   # ① 자료 수집
    cutoff = datetime.now(timezone.utc) - timedelta(hours=s["hours"])
    items, dead, seen = [], [], set()
    for name, url in SOURCES:
        try:
            feed = feedparser.parse(requests.get(url, headers=UA, timeout=20).content)
        except Exception:
            dead.append(name)                   # 한 곳이 죽어도 나머지는 계속
            continue
        for e in feed.entries:
            at = published_at(e)
            if not at or at < cutoff:           # time window 밖이거나 날짜가 없으면 버린다
                continue
            if name in KEYWORD_FILTER:                  # 클래식과 무관한 글은 버린다
                text = e.get("title", "") + " " + strip_tags(e.get("summary", ""))
                if not any(k in text for k in KEYWORD_FILTER[name]):
                    continue
            key = e.link.split("?")[0].rstrip("/")       # 추적용 꼬리표를 떼고 비교
            if key in seen:
                continue
            seen.add(key)
            items.append({"title": e.title, "url": e.link, "source": name, "at": at,
                          "summary": strip_tags(e.get("summary", ""))[:300]})
    return {"collected": items,
            "log": [f"① 수집   {s['hours']}시간 time window · {len(items)}건"
                    + (f" · 응답 없음 {dead}" if dead else "")]}

# ── ② select ─────────────────────────────────────────────────────────
class Pick(BaseModel):
    index: int = Field(description="후보 목록에서의 번호")
    reason: str = Field(description="왜 골랐는지 한 문장")
    event: str = Field(description="이 기사가 다루는 사건을 짧은 라벨로. 같은 사건이면 같은 라벨")

class Shortlist(BaseModel):
    picks: list[Pick]

BATCH, TARGET = 40, 5          # 예선 묶음 크기, 최종 발행 건수
CRITERIA = ("독자는 클래식 공연을 직접 보러 가는 국내 클래식 팬입니다.\n"
            "우선순위는 다음과 같습니다.\n"
            "1. 피아니스트 임윤찬·조성진(Yunchan Lim, Seong-Jin Cho)의 공연 일정·음반·활동 소식\n"
            "2. 오케스트라 협연, 내한 공연, 시즌 프로그램 등 앞으로 볼 수 있는 공연 소식\n"
            "3. 클래식 업계의 주요 동향(콩쿠르, 음반, 단체·인사 소식)\n"
            "버릴 것: 이미 끝난 공연의 단순 후기, 클래식과 무관한 문화 소식, 광고성 소식,\n"
            "지자체·학교·수목원 등 지역 소규모 행사, 연주자의 사건·사고·가십")

def ask_picks(items, n):
    listing = "\n".join(f"{i}. [{it['source']}] {it['title']}" for i, it in enumerate(items))
    sys = (f"{CRITERIA}\n\n아래 목록에서 중요한 순서대로 {n}건을 고르세요.\n"
           "같은 사건을 다룬 기사에는 같은 event 라벨을 붙이세요.")
    out = client.chat.completions.parse(
        model="gpt-4.1-mini", temperature=0,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": listing}],
        response_format=Shortlist).choices[0].message.parsed
    return [p for p in out.picks if 0 <= p.index < len(items)]      # 없는 번호는 버린다

# 이 이름이 제목·요약에 있으면 예선·본선을 건너뛰고 맨 앞에 넣는다(코드로 강제)
PIN_KEYS = ["임윤찬", "조성진", "yunchan lim", "lim yunchan", "seong-jin cho", "seong jin cho",
            "cho seong-jin", "cho seong jin"]

def is_pinned(it):
    text = (it["title"] + " " + it["summary"]).lower()
    return any(k in text for k in PIN_KEYS)

def select(s: dict) -> dict:                   # ② 중요도 선별
    items = s["collected"]
    pinned = [it for it in items if is_pinned(it)]
    rest = [it for it in items if not is_pinned(it)]
    top, seen_event = [], set()
    if pinned:                                  # 고정 기사끼리는 순위만 매기고, 같은 사건은 한 건만
        for p in ask_picks(pinned, min(len(pinned), TARGET * 2)):
            if p.event in seen_event:
                continue
            seen_event.add(p.event)
            top.append(pinned[p.index])
        top = top[:TARGET]
    need = TARGET - len(top)
    survivors, extra = [], []
    if need > 0 and rest:
        for i in range(0, len(rest), BATCH):    # 예선 — 묶음마다 여덟 건
            chunk = rest[i:i + BATCH]
            survivors += [chunk[p.index] for p in ask_picks(chunk, 8)]
        finals = ask_picks(survivors, need)     # 본선 — 한 화면에 놓고 남은 자리만큼
        extra = [survivors[p.index] for p in finals]
    return {"picked": top + extra,
            "log": [f"② 선별   {len(items)} → 고정 {len(top)} + 예선 {len(survivors)} → {len(top) + len(extra)}건"]}

# ── ③ report ─────────────────────────────────────────────────────────
class Draft(BaseModel):        # 22장에서 정한 세 칸
    headline: str = Field(description="20자 내외의 한국어 헤드라인")
    summary:  str = Field(description="세 문장 요약. ~합니다체, 과장 없이 건조하게")
    why:      str = Field(description="클래식 공연을 보러 가는 국내 팬에게 왜 중요한지 한 문장")

SYS = ("당신은 국내 클래식 팬을 위한 뉴스레터 기자입니다.\n"
       "아래 기사 본문(영어·독일어·한국어일 수 있음)을 읽고 헤드라인·요약·왜 중요한지를 한국어로 쓰세요.\n"
       "공연 날짜·장소·연주자·곡목이 본문에 있으면 요약에 정확히 넣고, 본문에 없는 정보는 절대 지어내지 마세요.\n"
       "'주목된다·기대를 모은다' 같은 기자체 표현은 쓰지 마세요.")

class ReportIn(TypedDict):                     # 워커가 받는 것은 기사 하나뿐
    item: dict

def extract_body(url):
    d = trafilatura.fetch_url(url)
    return trafilatura.extract(d) if d else None

def draft(body):
    return client.chat.completions.parse(
        model="gpt-4.1-mini", temperature=0,
        messages=[{"role": "system", "content": SYS},
                  {"role": "user", "content": body[:6000]}],
        response_format=Draft).choices[0].message.parsed

def fan_report(s: dict):                       # 기사 수만큼 워커를 펼친다
    return [Send("report", {"item": it}) for it in s["picked"]]

def report(s: ReportIn) -> dict:               # ③ 요약 — 기사 한 건을 맡는다
    it = s["item"]
    body = extract_body(it["url"])
    if not body or len(body) < 600:            # 11장에서 정한 G1 기준선
        return {"drafted": [],
                "log": [f"   취재 제외 {it['source']} · 본문 {len(body or '')}자"]}
    d = draft(body)
    return {"drafted": [{**it, "body": body[:6000], **d.model_dump()}]}

# ── ④ verify ─────────────────────────────────────────────────────────
class Verdict(BaseModel):
    ok:       bool      = Field(description="요약이 원문에 근거하면 true")
    problems: list[str] = Field(description="근거 없는 부분. 없으면 빈 목록")

SYS_CHECK = ("요약이 원문에서 뒷받침되는지 판정하세요.\n"
             "헤드라인과 요약만 보고 판단하고, 번역이나 단위 환산은 문제가 아닙니다.")

def check(d):                                  # 요약을 쓴 호출과 따로 부른다
    user = (f"[원문]\n{d['body'][:5000]}\n\n"
            f"[헤드라인]\n{d['headline']}\n\n[요약]\n{d['summary']}")
    return client.chat.completions.parse(
        model="gpt-4.1-mini", temperature=0,
        messages=[{"role": "system", "content": SYS_CHECK},
                  {"role": "user", "content": user}],
        response_format=Verdict).choices[0].message.parsed

def verify(s: dict) -> dict:                   # ④ 검수
    kept, dropped = [], []
    for d in s["drafted"]:
        (kept if check(d).ok else dropped).append(d)
    return {"verified": kept,                  # 리듀서 없는 새 칸에 담는다
            "log": [f"④ 검수   {len(s['drafted'])} → {len(kept)}건"
                    + (f" · 불합격 {[x['source'] for x in dropped]}" if dropped else "")]}

# ── ⑤ publish ────────────────────────────────────────────────────────
COLORS = {"모델·API": 0x0B6E77, "도구·프레임워크": 0x4C7C9C,
          "정책·규제": 0x8F5606, "사례·적용": 0x2E7D5B, "연구": 0x6A4A9C}
DEFAULT = 0x5F7476
TITLE_MAX, DESC_MAX, EMBED_MAX, TOTAL_MAX = 256, 4096, 10, 5800   # 6000에서 여유를 둔다

def build_embeds(run_id, lead, articles):
    if not articles:                                   # 조용한 날에도 한 장은 보낸다
        return [{"title": f"🗞️ {run_id}", "color": DEFAULT,
                 "description": "오늘은 조용합니다."}]
    embeds = [{"title": f"🎹 {run_id} · 클래식 브리핑", "description": lead, "color": DEFAULT}]
    for i, a in enumerate(articles, 1):
        desc = a["summary"]
        if a.get("why"):
            desc += f"\n\n💡 **{a['why']}**"
        embeds.append({
            "title":       f"{i}. {a['headline']}"[:TITLE_MAX],
            "description": desc[:DESC_MAX],
            "url":         a["url"],
            "color":       COLORS.get(a.get("topic", ""), DEFAULT),
            "footer":      {"text": f"{a['source']} · {a['when']}"},
        })
    total = lambda es: sum(len(e.get("title", "")) + len(e.get("description", ""))
                           + len(e.get("footer", {}).get("text", "")) for e in es)
    while len(embeds) > EMBED_MAX or total(embeds) > TOTAL_MAX:
        embeds.pop()                                   # 뒤에서부터 덜어낸다
    return embeds

def send(run_id, lead, articles, webhook=None, dry_run=True):
    payload = {"username": "편집실", "embeds": build_embeds(run_id, lead, articles)}
    if dry_run or not webhook:
        print(f"[dry-run] embed {len(payload['embeds'])}개 · "
              f"{len(json.dumps(payload, ensure_ascii=False))}자 — 보내지 않음")
        print(json.dumps(payload["embeds"][-1], ensure_ascii=False, indent=2)[:400])
        return False
    r = requests.post(webhook, json=payload, timeout=20)
    ok = r.status_code in (200, 204)
    print("발행:", "성공" if ok else f"실패 {r.status_code} {r.text[:120]}")
    return ok

def make_lead(arts):
    if not arts:
        return ""
    srcs = ", ".join(dict.fromkeys(a["source"] for a in arts))
    return f"오늘은 {len(arts)}건을 골랐습니다. ({srcs})"

def publish(s: dict) -> dict:                 # ⑤ 발행
    arts = [{"headline": a["headline"], "summary": a["summary"], "why": a["why"],
             "url": a["url"], "source": a["source"], "topic": a.get("event", ""),
             "when": a["at"].strftime("%m-%d %H:%M")} for a in s["verified"]]
    today = datetime.now().strftime("%Y-%m-%d")
    sent  = send(today, make_lead(arts), arts,
                 webhook=os.environ.get("DISCORD_WEBHOOK_URL"),
                 dry_run=os.environ.get("DRY_RUN", "1") == "1")   # 기본은 보내지 않음
    label = f"{len(arts)}건" if arts else "조용합니다"
    return {"log": [f"⑤ 발행   {label} · {'보냄' if sent else 'dry-run'}"]}

# ── 그래프 ───────────────────────────────────────────────────────────
def build():                                   # 노드를 이름으로 찾아 붙인다
    g = StateGraph(Brief)
    for name in ("collect", "select", "report", "verify", "publish"):
        g.add_node(name, globals()[name])
    g.add_edge(START, "collect")
    g.add_edge("collect", "select")
    g.add_conditional_edges("select", fan_report, ["report"])   # 기사 수만큼 팬아웃
    g.add_edge("report", "verify")
    g.add_edge("verify", "publish")
    g.add_edge("publish", END)
    return g

def run():
    return build().compile().invoke(INIT)
