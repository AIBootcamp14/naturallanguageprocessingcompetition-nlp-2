# src/data/preprocess/denoise.py
import os
import re
import pandas as pd
from typing import Iterable

# ===== 규칙 세트 =====
ALLOWED_MASKS = {
    "Address", "CarNumber", "CardNumber", "DateOfBirth", "Email", "FlightNumber",
    "Name", "PassportNumber", "Person1", "Person2", "Person3", "Person4", "Person5",
    "Person6", "Person7", "PersonName", "PhoneNumber", "Price", "SSN"
}

RX = {
    # #...# 스팬
    "SPAN": re.compile(r"#(.*?)#"),
    # #MASK# + 숫자 / 숫자 + #MASK# (구분자 허용)
    "ATTACH_POST": re.compile(r"(#\w+#)(\d+)"),
    "ATTACH_PRE":  re.compile(r"(\d+)(#\w+#)"),
    # 제목 괄호 <...> → '...'
    "TITLE_ANGLE": re.compile(r"<([^<>]+)>"),
    # (한자)(영문로마자) → 영문로마자만
    "HANJA_ROMA": re.compile(r"([\u4E00-\u9FFF]+)\s*\(([A-Za-z][^)]*)\)"),
    # $#,000 류(‘#’가 포함된 경우만 치환)
    "CURRENCY_HASHED": re.compile(r"\$[#,0-9][#,0-9\.]*"),
}


def _strip_nonmask_hashtags(text: str) -> str:
    """화이트리스트 외 #...# 는 # 제거(내용만 남김). 나머지는 유지."""
    if not isinstance(text, str):
        return text

    def repl(m):
        inner = m.group(1).strip()
        return f"#{inner}#" if inner in ALLOWED_MASKS else inner
    return RX["SPAN"].sub(repl, text)


def _normalize_common(text: str) -> str:
    """공통 정규화(대화/요약 공통)."""
    if not isinstance(text, str):
        return text
    # 줄바꿈 통일
    text = (text.replace("\\n", "\n")
                .replace("<br>", "\n").replace("<BR>", "\n").replace("<Br>", "\n"))
    # 따옴표/괄호 통일
    text = (text.replace("‘", "'").replace("’", "'")
                .replace("『", "'").replace("』", "'")
                .replace("《", "'").replace("》", "'"))
    # 제목 괄호 <...> → '...'
    text = RX["TITLE_ANGLE"].sub(r"'\1'", text)
    # 대시 계열 통일
    text = text.replace("—", "-").replace("―", "-")
    # 역슬래시 → 슬래시
    text = text.replace("\\", "/")
    # 가운데점 → 쉼표
    text = text.replace("·", ", ")
    # 유니코드 말줄임표 → ASCII
    text = text.replace("…", "...")
    # 감정 표현(희소 케이스만)
    text = re.sub(r"(?<![가-힣])[ㅎㅋ]{2,}(?![가-힣])", "하하", text)   # ㅎㅎ, ㅋㅋ
    text = re.sub(r"[ㅠㅜ]{2,}", "흑흑", text)                         # ㅠㅠ, ㅜㅜ
    # 한자( )영문 → 영문만
    text = RX["HANJA_ROMA"].sub(r"\2", text)
    return text


def _detach_numbers_from_masks(text: str) -> str:
    """#MASK#25395 → #MASK#, 021-#MASK# → #MASK#"""
    text = RX["ATTACH_POST"].sub(r"\1", text)
    text = RX["ATTACH_PRE"].sub(r"\2", text)
    return text


def _remove_orphan_hash(text: str) -> str:
    if not isinstance(text, str):
        return text
    protected = []

    def protect(m):
        protected.append(m.group(0))
        return f"@@MASK{len(protected)-1}@@"
    t = RX["SPAN"].sub(protect, text)  # 마스크 보호
    t = t.replace("#", "")             # 고립 # 제거
    for i, orig in enumerate(protected):
        t = t.replace(f"@@MASK{i}@@", orig)  # 복원
    return t


def _currency_hashtag_to_amount(text: str, apply_amount_mask: bool) -> str:
    """$#,000 → #Amount#  (apply_amount_mask=False면 건드리지 않음)"""
    if not apply_amount_mask:
        return text

    def repl(m):
        tok = m.group(0)
        return "#Amount#" if "#" in tok else tok
    return RX["CURRENCY_HASHED"].sub(repl, text)


def denoise_text(text: str, *, column: str = "dialogue") -> str:
    """열별 규칙 포함한 전체 파이프라인."""
    if not isinstance(text, str):
        return text
    # 1) 화이트리스트 외 #...# 해제
    text = _strip_nonmask_hashtags(text)
    # 2) 공통 정규화
    text = _normalize_common(text)
    # 3) #MASK#±숫자 분리
    text = _detach_numbers_from_masks(text)
    # 4) 곱하기 기호 → 곱하기
    text = re.sub(r'(?<=\d)\*(?=\d)', ' 곱하기 ', text)
    # 5) 통화 해시 금액 처리(대화에서만)
    text = _currency_hashtag_to_amount(
        text, apply_amount_mask=(column == "dialogue"))
    # 6) 고립 # 제거
    text = _remove_orphan_hash(text)
    return text


def denoise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dialogue" in out.columns:
        out["dialogue"] = out["dialogue"].apply(
            lambda x: denoise_text(x, column="dialogue"))
    if "summary" in out.columns:
        out["summary"] = out["summary"].apply(
            lambda x: denoise_text(x, column="summary"))
    return out


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def run(input_paths: Iterable[str], output_dir: str = "data/clean/denoise"):
    _ensure_dir(output_dir)
    for ip in input_paths:
        df = pd.read_csv(ip)
        cleaned = denoise_dataframe(df)
        fname = os.path.basename(ip)
        cleaned.to_csv(os.path.join(output_dir, fname),
                       index=False, encoding="utf-8")


if __name__ == "__main__":
    run(["data/train.csv", "data/dev.csv", "data/test.csv"])
