"""Build a Korean, chart-first educational PDF for decoding trading signals.

The document is deliberately channel-independent: no Telegram message, provider,
or historical signal outcome is asserted without a source export and timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

from reportlab.lib import colors
from reportlab.lib.colors import Color
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT: Final = Path(__file__).resolve().parents[1]
OUTPUT: Final = ROOT / "output" / "pdf" / "altcoin-chart-entry-exit-playbook-ko.pdf"
FONT_REGULAR: Final = Path(r"C:\Windows\Fonts\malgun.ttf")
FONT_BOLD: Final = Path(r"C:\Windows\Fonts\malgunbd.ttf")
PAGE_W, PAGE_H = A4
NAVY: Final = Color(0.045, 0.11, 0.20)
INK: Final = Color(0.08, 0.13, 0.19)
MUTED: Final = Color(0.34, 0.40, 0.48)
PALE: Final = Color(0.94, 0.97, 0.99)
BLUE: Final = Color(0.10, 0.40, 0.80)
TEAL: Final = Color(0.00, 0.59, 0.53)
ORANGE: Final = Color(0.93, 0.43, 0.15)
RED: Final = Color(0.80, 0.20, 0.24)
GRID: Final = Color(0.84, 0.88, 0.92)


def _register_fonts() -> None:
    if not FONT_REGULAR.exists() or not FONT_BOLD.exists():
        raise FileNotFoundError("Malgun Gothic fonts are required to render Korean text.")
    pdfmetrics.registerFont(TTFont("Malgun", str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont("MalgunBold", str(FONT_BOLD)))


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleKo",
            parent=base["Title"],
            fontName="MalgunBold",
            fontSize=26,
            leading=35,
            textColor=colors.white,
            alignment=TA_LEFT,
            wordWrap="CJK",
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleKo",
            parent=base["Normal"],
            fontName="Malgun",
            fontSize=11.5,
            leading=18,
            textColor=Color(0.80, 0.87, 0.94),
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "H1Ko",
            parent=base["Heading1"],
            fontName="MalgunBold",
            fontSize=19,
            leading=27,
            textColor=NAVY,
            spaceBefore=4,
            spaceAfter=11,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "H2Ko",
            parent=base["Heading2"],
            fontName="MalgunBold",
            fontSize=13.2,
            leading=19,
            textColor=INK,
            spaceBefore=10,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "BodyKo",
            parent=base["BodyText"],
            fontName="Malgun",
            fontSize=9.6,
            leading=16,
            textColor=INK,
            wordWrap="CJK",
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallKo",
            parent=base["BodyText"],
            fontName="Malgun",
            fontSize=7.7,
            leading=11.5,
            textColor=MUTED,
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "TableKo",
            parent=base["BodyText"],
            fontName="Malgun",
            fontSize=8.1,
            leading=12.2,
            textColor=INK,
            wordWrap="CJK",
        ),
        "table_head": ParagraphStyle(
            "TableHeadKo",
            parent=base["BodyText"],
            fontName="MalgunBold",
            fontSize=8.2,
            leading=12,
            textColor=colors.white,
            wordWrap="CJK",
            alignment=TA_CENTER,
        ),
        "callout": ParagraphStyle(
            "CalloutKo",
            parent=base["BodyText"],
            fontName="Malgun",
            fontSize=9.6,
            leading=15.5,
            textColor=INK,
            wordWrap="CJK",
        ),
        "cover_note": ParagraphStyle(
            "CoverNoteKo",
            parent=base["BodyText"],
            fontName="Malgun",
            fontSize=8.5,
            leading=13,
            textColor=Color(0.69, 0.78, 0.86),
            wordWrap="CJK",
        ),
    }


class Cover(Flowable):
    def __init__(self, styles: dict[str, ParagraphStyle]) -> None:
        super().__init__()
        self.width = PAGE_W - 32 * mm
        self.height = 245 * mm
        self.styles = styles

    def draw(self) -> None:
        c = self.canv
        c.setFillColor(NAVY)
        c.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        c.setFillColor(Color(0.06, 0.26, 0.43))
        c.circle(self.width - 17 * mm, self.height - 38 * mm, 52 * mm, fill=1, stroke=0)
        c.setFillColor(Color(0.08, 0.40, 0.62))
        c.circle(self.width - 24 * mm, self.height - 35 * mm, 35 * mm, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.rect(18 * mm, self.height - 45 * mm, 14 * mm, 3 * mm, fill=1, stroke=0)

        y = self.height - 70 * mm
        title = Paragraph("알트코인 차트에서<br/>진입 · 퇴장 · TP/SL을<br/>조건부로 판독하는 법", self.styles["title"])
        title.wrapOn(c, self.width - 36 * mm, 100 * mm)
        title.drawOn(c, 18 * mm, y)
        subtitle = Paragraph(
            "텔레그램 신호를 숫자만 따라 적지 않고, 신호가 성립하는 시장 상태와\n"
            "무효화 조건을 직접 차트에서 재현하기 위한 교육용 프레임워크",
            self.styles["subtitle"],
        )
        subtitle.wrapOn(c, self.width - 36 * mm, 45 * mm)
        subtitle.drawOn(c, 18 * mm, 105 * mm)
        c.setStrokeColor(Color(0.35, 0.52, 0.66))
        c.setLineWidth(0.5)
        c.line(18 * mm, 89 * mm, self.width - 18 * mm, 89 * mm)
        note = Paragraph(
            "2026-09-04 · 공공 차트 구조와 거래소 운영 문서를 바탕으로 한 교육 자료\n"
            "특정 채널의 신호 원문, 적중률, 수익을 주장하거나 매매를 권유하지 않습니다.",
            self.styles["cover_note"],
        )
        note.wrapOn(c, self.width - 36 * mm, 30 * mm)
        note.drawOn(c, 18 * mm, 65 * mm)
        c.setFont("MalgunBold", 8)
        c.setFillColor(Color(0.63, 0.75, 0.84))
        c.drawString(18 * mm, 23 * mm, "CHART READING PLAYBOOK · EDUCATIONAL USE")


@dataclass(frozen=True, slots=True)
class Candle:
    open_value: float
    close_value: float
    high: float
    low: float


class ChartScenario(Flowable):
    """Compact schematic candlestick chart. It is intentionally not market data."""

    def __init__(self, title: str, mode: str, caption: str) -> None:
        super().__init__()
        self.width = 170 * mm
        self.height = 76 * mm
        self.title = title
        self.mode = mode
        self.caption = caption

    def _candles(self) -> tuple[Candle, ...]:
        if self.mode == "long_pullback":
            values = (88, 92, 95, 98, 96, 93, 90, 86, 91, 95, 97, 100, 104, 108)
        elif self.mode == "short_pullback":
            values = (110, 106, 102, 98, 101, 105, 109, 112, 107, 103, 99, 96, 92, 88)
        elif self.mode == "breakout":
            values = (88, 89, 90, 89, 91, 90, 92, 97, 101, 98, 102, 106, 110, 113)
        else:
            values = (100, 103, 105, 104, 102, 99, 96, 94, 97, 100, 102, 101, 99, 98)
        result: list[Candle] = []
        for index, close_value in enumerate(values):
            open_value = values[index - 1] if index else values[0] - 2
            wick = 1.3 + (index % 3) * 0.55
            result.append(Candle(open_value, close_value, max(open_value, close_value) + wick, min(open_value, close_value) - wick))
        return tuple(result)

    def _price_y(self, value: float, floor: float, ceiling: float, bottom: float, chart_h: float) -> float:
        return bottom + ((value - floor) / (ceiling - floor)) * chart_h

    def draw(self) -> None:
        c = self.canv
        left, bottom, chart_w, chart_h = 11 * mm, 17 * mm, 148 * mm, 43 * mm
        candles = self._candles()
        values = [v for candle in candles for v in (candle.high, candle.low)]
        floor, ceiling = min(values) - 2, max(values) + 2
        c.setFillColor(PALE)
        c.roundRect(0, 0, self.width, self.height, 3 * mm, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont("MalgunBold", 9)
        c.drawString(left, self.height - 10 * mm, self.title)
        c.setStrokeColor(GRID)
        c.setLineWidth(0.35)
        for fraction in (0.2, 0.5, 0.8):
            y = bottom + chart_h * fraction
            c.line(left, y, left + chart_w, y)
        c.setStrokeColor(Color(0.65, 0.72, 0.78))
        c.line(left, bottom, left, bottom + chart_h)
        c.line(left, bottom, left + chart_w, bottom)
        spacing = chart_w / len(candles)
        for index, candle in enumerate(candles):
            x = left + spacing * (index + 0.5)
            y_open = self._price_y(candle.open_value, floor, ceiling, bottom, chart_h)
            y_close = self._price_y(candle.close_value, floor, ceiling, bottom, chart_h)
            y_high = self._price_y(candle.high, floor, ceiling, bottom, chart_h)
            y_low = self._price_y(candle.low, floor, ceiling, bottom, chart_h)
            color = TEAL if candle.close_value >= candle.open_value else RED
            c.setStrokeColor(color)
            c.setFillColor(color)
            c.setLineWidth(0.8)
            c.line(x, y_low, x, y_high)
            c.rect(x - 1.55 * mm, min(y_open, y_close), 3.1 * mm, max(abs(y_close - y_open), 0.8 * mm), fill=1, stroke=0)

        labels: dict[str, tuple[float, float, Color]]
        if self.mode == "long_pullback":
            labels = {
                "4H 지지대": (91, BLUE),
                "스윕 후 회복": (95, TEAL),
                "진입": (95, TEAL),
                "무효화 SL": (84, RED),
                "TP1 / TP2": (104, ORANGE),
            }
            x_points = {"4H 지지대": 6.3, "스윕 후 회복": 8.4, "진입": 9.3, "무효화 SL": 7.2, "TP1 / TP2": 13.1}
        elif self.mode == "short_pullback":
            labels = {
                "4H 저항": (109, BLUE),
                "스윕 후 재하락": (106, TEAL),
                "진입": (105, TEAL),
                "무효화 SL": (114, RED),
                "TP1 / TP2": (94, ORANGE),
            }
            x_points = {"4H 저항": 6.2, "스윕 후 재하락": 8.5, "진입": 9.2, "무효화 SL": 7.2, "TP1 / TP2": 12.9}
        elif self.mode == "breakout":
            labels = {
                "기존 상단": (92, BLUE),
                "종가 돌파": (98, TEAL),
                "리테스트 진입": (98, TEAL),
                "무효화 SL": (94, RED),
                "다음 유동성": (109, ORANGE),
            }
            x_points = {"기존 상단": 5.2, "종가 돌파": 8.4, "리테스트 진입": 10.1, "무효화 SL": 9.4, "다음 유동성": 13.1}
        else:
            labels = {
                "레인지 상단": (105, BLUE),
                "중앙선은 관망": (100, MUTED),
                "레인지 하단": (95, BLUE),
                "반전 확인": (98, TEAL),
                "무효화 SL": (92, RED),
            }
            x_points = {"레인지 상단": 2.7, "중앙선은 관망": 7.5, "레인지 하단": 8.0, "반전 확인": 9.1, "무효화 SL": 7.2}
        for label, (price, color) in labels.items():
            y = self._price_y(price, floor, ceiling, bottom, chart_h)
            c.setStrokeColor(color)
            c.setDash(2, 2)
            c.line(left, y, left + chart_w, y)
            c.setDash()
            x = left + spacing * x_points[label]
            c.setFillColor(color)
            c.setFont("MalgunBold", 6.8)
            c.drawString(min(x, left + chart_w - 22 * mm), min(y + 1.8 * mm, bottom + chart_h - 1.5 * mm), label)
        c.setFillColor(MUTED)
        c.setFont("Malgun", 7.1)
        c.drawString(left, 6 * mm, self.caption)


class Timeline(Flowable):
    def __init__(self) -> None:
        super().__init__()
        self.width = 170 * mm
        self.height = 54 * mm

    def draw(self) -> None:
        c = self.canv
        c.setFillColor(PALE)
        c.roundRect(0, 0, self.width, self.height, 3 * mm, fill=1, stroke=0)
        items = (
            ("1D / 4H", "방향과 주요 구간", NAVY),
            ("1H / 15m", "셋업과 구조 변화", BLUE),
            ("5m / 1m", "진입 촉발과 체결", TEAL),
        )
        box_w, box_h, gap = 47 * mm, 28 * mm, 6 * mm
        for index, (timeframe, description, color) in enumerate(items):
            x = 8 * mm + index * (box_w + gap)
            y = 15 * mm
            c.setFillColor(colors.white)
            c.setStrokeColor(color)
            c.setLineWidth(1.0)
            c.roundRect(x, y, box_w, box_h, 2 * mm, fill=1, stroke=1)
            c.setFillColor(color)
            c.setFont("MalgunBold", 10)
            c.drawCentredString(x + box_w / 2, y + 17 * mm, timeframe)
            c.setFillColor(INK)
            c.setFont("Malgun", 7.5)
            c.drawCentredString(x + box_w / 2, y + 9 * mm, description)
            if index < len(items) - 1:
                arrow_x = x + box_w + 1.5 * mm
                c.setStrokeColor(MUTED)
                c.line(arrow_x, y + box_h / 2, arrow_x + 3 * mm, y + box_h / 2)
                c.setFillColor(MUTED)
                c.translate(arrow_x + 3 * mm, y + box_h / 2)
                c.rotate(180)
                c.line(0, 0, 2 * mm, 1.5 * mm)
                c.line(0, 0, 2 * mm, -1.5 * mm)
                c.rotate(-180)
                c.translate(-(arrow_x + 3 * mm), -(y + box_h / 2))
        c.setFillColor(MUTED)
        c.setFont("Malgun", 7.2)
        c.drawString(8 * mm, 6 * mm, "상위 시간봉이 반대면 하위 시간봉의 예쁜 촉발도 역추세 반등일 수 있다.")


class DecisionFlow(Flowable):
    def __init__(self) -> None:
        super().__init__()
        self.width = 170 * mm
        self.height = 80 * mm

    def draw(self) -> None:
        c = self.canv
        c.setFillColor(PALE)
        c.roundRect(0, 0, self.width, self.height, 3 * mm, fill=1, stroke=0)
        steps = (
            ("1", "상위 구조", "상승 · 하락 · 레인지"),
            ("2", "가격 위치", "지지/저항 · 유동성 · 중간값"),
            ("3", "변동성", "정상 · 확장 · 비정상 스파이크"),
            ("4", "하위 촉발", "스윕 · 종가 회복 · 리테스트"),
            ("5", "계획", "진입 · 무효화 · TP · 수량"),
        )
        box_w, box_h = 29 * mm, 31 * mm
        y = 30 * mm
        for index, (number, title, detail) in enumerate(steps):
            x = 7 * mm + index * 33 * mm
            color = (NAVY, BLUE, ORANGE, TEAL, RED)[index]
            c.setFillColor(colors.white)
            c.setStrokeColor(color)
            c.roundRect(x, y, box_w, box_h, 2 * mm, fill=1, stroke=1)
            c.setFillColor(color)
            c.circle(x + 5 * mm, y + 24 * mm, 3.5 * mm, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont("MalgunBold", 7)
            c.drawCentredString(x + 5 * mm, y + 22.6 * mm, number)
            c.setFillColor(INK)
            c.setFont("MalgunBold", 7.2)
            c.drawCentredString(x + box_w / 2, y + 16 * mm, title)
            c.setFont("Malgun", 5.8)
            c.drawCentredString(x + box_w / 2, y + 9 * mm, detail)
            if index < len(steps) - 1:
                c.setStrokeColor(MUTED)
                c.line(x + box_w, y + box_h / 2, x + 33 * mm - 1.3 * mm, y + box_h / 2)
        c.setFillColor(RED)
        c.setFont("MalgunBold", 8)
        c.drawCentredString(self.width / 2, 14 * mm, "하나라도 '아니오'이면 보류 - 억지로 진입 신호를 만들지 않는다.")


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def callout(text: str, styles: dict[str, ParagraphStyle], color: Color = PALE) -> Table:
    table = Table([[p(text, styles["callout"])]], colWidths=[170 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("BOX", (0, 0), (-1, -1), 0.55, Color(0.73, 0.80, 0.86)),
                ("LEFTPADDING", (0, 0), (-1, -1), 5 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5 * mm),
            ]
        )
    )
    return table


def rule_table(headers: Sequence[str], rows: Sequence[Sequence[str]], styles: dict[str, ParagraphStyle], widths: Sequence[float]) -> Table:
    data: list[list[Paragraph]] = [[p(header, styles["table_head"]) for header in headers]]
    for row in rows:
        data.append([p(cell, styles["table"]) for cell in row])
    table = Table(data, colWidths=list(widths), repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, Color(0.79, 0.84, 0.89)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.5 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.5 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2.2 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2 * mm),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, Color(0.975, 0.985, 0.995)]),
            ]
        )
    )
    return table


def section(title: str, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    return [p(title, styles["h1"])]


def body(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return p(text, styles["body"])


def footer(canvas, doc) -> None:  # type: ignore[no-untyped-def]
    page = canvas.getPageNumber()
    if page == 1:
        return
    canvas.saveState()
    canvas.setStrokeColor(Color(0.81, 0.86, 0.90))
    canvas.line(16 * mm, 13 * mm, PAGE_W - 16 * mm, 13 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Malgun", 7.2)
    canvas.drawString(16 * mm, 8.2 * mm, "알트코인 차트 판독 플레이북 · 교육용")
    canvas.drawRightString(PAGE_W - 16 * mm, 8.2 * mm, str(page))
    canvas.restoreState()


def build() -> Path:
    _register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=19 * mm,
        title="알트코인 차트 진입 퇴장 TP SL 판독 플레이북",
        author="Codex - educational research artifact",
    )
    story: list[Flowable] = [Cover(styles), PageBreak()]

    story += section("1. 이 문서가 가르치는 것", styles)
    story += [
        body(
            "차트에서 '언제 들어가고 언제 나올지'를 맞히는 일은 한 개의 지표나 한 번의 캔들 모양을 찾는 일이 아니다. "
            "좋은 신호는 <b>시장 상태 → 가격 위치 → 촉발 → 무효화 → 목표</b>가 한 문장으로 이어진다. "
            "이 문서는 텔레그램의 숫자를 복사하는 대신, 그 문장이 성립하는지 직접 판독하는 순서를 제공한다.",
            styles,
        ),
        callout(
            "<b>핵심 원칙:</b> 진입은 '오를 것 같다'가 아니라, 정해 둔 무효화 가격까지의 손실을 감수할 이유가 있고 "
            "그보다 가까운 반대 유동성 또는 구조 목표가 충분히 멀 때만 검토한다. 손절은 예측이 틀렸다는 증거이고, TP는 "
            "예측이 맞아도 반대 주문이 몰릴 수 있는 구간이다.",
            styles,
        ),
        Spacer(1, 5 * mm),
        rule_table(
            ("구성 요소", "차트에서 보는 것", "질문"),
            (
                ("상태", "4H/1H 추세, 레인지, 변동성", "추세 추종인가, 되돌림인가, 관망인가?"),
                ("위치", "이전 고점·저점, 박스 상·하단, 갭/급등락 시작점", "가격이 가운데인가, 의사결정 구간인가?"),
                ("촉발", "스윕 뒤 종가 회복, 종가 돌파, 리테스트 실패/성공", "지금이 아니라면 무엇이 확인되면 들어갈까?"),
                ("무효화", "구조가 깨지는 가격과 시간", "이 가격을 넘으면 내 가설은 왜 틀렸나?"),
                ("목표", "반대 유동성, 이전 확장, R 배수", "어디서 일부 청산하고, 어디서 전량 퇴장할까?"),
            ),
            styles,
            (27 * mm, 66 * mm, 77 * mm),
        ),
        Spacer(1, 4 * mm),
        body(
            "<b>한계:</b> 이 문서는 특정 텔레그램 채널의 원문·게시 시각·수정 이력·실제 체결을 받지 않았으므로, 해당 채널의 "
            "고유 규칙이나 적중률을 주장하지 않는다. 아래의 규칙은 조건부 판독 도구이며 수익 보장이나 매매 권유가 아니다.",
            styles,
        ),
        PageBreak(),
    ]

    story += section("2. 여러 시간봉을 한 문장으로 연결하는 법", styles)
    story += [
        body(
            "시간봉을 많이 본다는 것은 1분봉부터 일봉까지 모두 같은 비중으로 본다는 뜻이 아니다. 상위 시간봉은 <b>어디서 거래할지</b>를, "
            "중간 시간봉은 <b>무슨 유형의 거래인지</b>를, 하위 시간봉은 <b>어떻게 체결할지</b>를 결정한다."
            " 하위 시간봉의 신호가 상위 시간봉의 위치와 충돌하면, 예쁜 5분봉 패턴도 역추세 반등일 수 있다.",
            styles,
        ),
        Timeline(),
        Spacer(1, 4 * mm),
        rule_table(
            ("차트", "반드시 표시할 것", "이 차트에서 하지 말 것"),
            (
                ("1D / 4H", "최근 두 개의 유의미한 고점·저점, 큰 박스 상·하단, 급등락 시작점", "1분봉 꼬리 하나로 방향을 뒤집지 않는다."),
                ("1H / 15m", "추세가 유지되는지, 구조가 바뀌는지, 진입 구간까지의 되돌림", "가격이 레인지 중간일 때 억지로 진입을 찾지 않는다."),
                ("5m / 1m", "스윕, 종가 회복/이탈, 리테스트, 체결 시점", "상위 시간봉 목표를 하위 시간봉 잡음 때문에 임의로 없애지 않는다."),
            ),
            styles,
            (25 * mm, 82 * mm, 63 * mm),
        ),
        Spacer(1, 4 * mm),
        callout(
            "<b>실전 문장 예:</b> '4H 상승 구조 안에서 1H 지지 구간으로 눌림이 왔고, 15m가 직전 저점을 잠깐 넘겼지만 종가가 다시 "
            "지지 위로 회복했다. 5m 리테스트가 버티면 롱을 검토한다. 15m 종가가 지지 아래에서 확정되면 가설은 무효다.' "
            "이 문장을 쓸 수 없으면 아직 진입 조건을 모르는 것이다.",
            styles,
            Color(0.94, 0.98, 0.97),
        ),
        PageBreak(),
    ]

    story += section("3. 차트를 보자마자 따르는 5단계 판독 순서", styles)
    story += [
        body(
            "아래 순서는 신호를 생산하는 공식이 아니라, 잘못된 신호를 제거하는 필터다. 한 단계가 불명확하면 다음 단계로 넘어가지 않는다. "
            "특히 가격이 레인지 중간이거나 급등락 뒤 과열 상태면 '아무 것도 하지 않음'도 완성된 판단이다.",
            styles,
        ),
        DecisionFlow(),
        Spacer(1, 4 * mm),
        rule_table(
            ("단계", "통과 기준", "보류 신호"),
            (
                ("1. 상위 구조", "HH/HL 또는 LH/LL, 혹은 명확한 박스", "고점·저점 정의가 시간봉마다 바뀜"),
                ("2. 가격 위치", "지지/저항, 박스 끝, 이전 유동성 근처", "가격이 박스 중앙 또는 이미 목표 바로 앞"),
                ("3. 변동성", "손절 폭과 목표 폭이 현재 ATR%에 비해 현실적", "한 캔들이 평소 범위를 여러 배 넘김"),
                ("4. 촉발", "종가 기준 회복/이탈 또는 리테스트 확인", "윗꼬리·아랫꼬리 한 번만 보고 진입"),
                ("5. 계획", "진입·SL·TP·수량·취소 조건을 먼저 적음", "진입 뒤에 손절과 목표를 고민"),
            ),
            styles,
            (28 * mm, 72 * mm, 70 * mm),
        ),
        PageBreak(),
    ]

    story += section("4. 먼저 분류해야 하는 네 가지 시장 상태", styles)
    story += [
        body(
            "같은 RSI, 이동평균, 캔들 패턴도 시장 상태가 바뀌면 의미가 달라진다. 추세장에서는 역추세 반전 신호가 빨리 무효화되고, "
            "레인지에서는 돌파 추종이 자주 되돌림을 맞는다. 따라서 '패턴'보다 먼저 '어떤 환경에서 이 패턴을 쓰는가'를 결정한다.",
            styles,
        ),
        rule_table(
            ("상태", "보이는 특징", "우선 검토", "피할 행동"),
            (
                ("상승/하락 추세", "고점·저점이 한 방향으로 갱신, 눌림이 얕거나 질서 있음", "추세 방향 눌림 진입, 돌파 리테스트", "반대 방향 저점/고점만 보고 계속 물타기"),
                ("레인지", "명확한 상단·하단, 중앙에서 왕복, 돌파 지속성 약함", "가장자리 반전, 상단/하단 이탈 후 확인", "중앙에서 R/R 없이 진입"),
                ("변동성 확장", "ATR% 상승, 캔들 몸통·꼬리 확대, 스윕 증가", "손절 폭 재계산, 체결 축소 또는 관망", "평소와 같은 좁은 손절, 추격 진입"),
                ("분배/전환", "상위 구조는 아직 유지되나 추진력이 약해지고 실패 돌파 반복", "구조 확인 뒤 전환 거래, 짧은 목표", "첫 반전 꼬리만 보고 큰 역추세 베팅"),
            ),
            styles,
            (28 * mm, 48 * mm, 48 * mm, 46 * mm),
        ),
        Spacer(1, 4 * mm),
        body(
            "ATR은 방향을 말해 주지 않고 변동성을 측정한다. 따라서 ATR/ATR%는 '롱인가 숏인가'의 근거가 아니라, 현재 구조적 손절이 "
            "너무 가까운지와 목표가 이미 소진됐는지를 점검하는 보조자다. <link href='https://www.tradingview.com/support/solutions/43000501823-average-true-range-atr/' color='blue'>TradingView ATR 설명</link>도 ATR이 방향 예측용 지표가 아니라고 명시한다.",
            styles,
        ),
        PageBreak(),
    ]

    story += section("5. 진입 유형 A: 추세 방향 눌림 - 롱", styles)
    story += [
        body(
            "가장 재현하기 쉬운 롱은 '가격이 올라가는 중이어서 산다'가 아니라, <b>상위 상승 구조 안에서 지지 구간으로 조정한 뒤 매도 "
            "유동성을 소화하고 다시 회복하는지</b>를 보는 거래다. 진입은 저점이 아니라 회복 증거가 나온 뒤에 한다.",
            styles,
        ),
        ChartScenario(
            "교육용 도식 - 상승 구조 안의 눌림 롱", "long_pullback",
            "도식일 뿐 실제 종목·시점·가격 데이터가 아니다. 종가 회복과 리테스트를 분리해 본다.",
        ),
        Spacer(1, 3 * mm),
        rule_table(
            ("순서", "차트 판독", "주문 계획으로 번역"),
            (
                ("1", "4H가 저점을 높이며 지지 구간 위에 있음", "롱만 우선 검토. 4H 반대 저항이 매우 가까우면 제외."),
                ("2", "1H/15m가 지지 안쪽 또는 직전 저점 아래를 잠깐 스윕", "스윕 그 자체는 진입 근거가 아니라 유동성 확인."),
                ("3", "15m 종가가 지지 위로 회복하고, 5m가 리테스트에서 버팀", "리테스트 종가 또는 확인된 제한가 구간을 진입 후보로 기록."),
                ("4", "회복 저점 아래로 재이탈", "그 가격과 ATR 여유를 반영한 곳이 무효화 SL 후보."),
                ("5", "직전 고점, 4H 반대 유동성, 확장 구간", "가장 가까운 목표부터 TP1, 다음 구조 목표를 TP2/잔여분 목표로."),
            ),
            styles,
            (12 * mm, 79 * mm, 81 * mm),
        ),
        PageBreak(),
    ]

    story += section("6. 진입 유형 B: 추세 방향 눌림 - 숏", styles)
    story += [
        body(
            "숏은 롱의 좌우 반전이지만, 알트코인은 급등 숏커버와 유동성 부족으로 위쪽 꼬리가 커질 수 있다. 따라서 '많이 올랐다'는 "
            "숏 근거가 아니다. 하락 구조 또는 상위 저항에서의 <b>실패 돌파 후 재하락</b>을 확인해야 한다.",
            styles,
        ),
        ChartScenario(
            "교육용 도식 - 하락 구조 안의 되돌림 숏", "short_pullback",
            "숏은 상단을 한 번 넘겼다는 사실보다, 그 위에서 종가를 유지하지 못했는지가 중요하다.",
        ),
        Spacer(1, 3 * mm),
        rule_table(
            ("확인", "숏이 가능한 해석", "숏을 미룰 해석"),
            (
                ("상위 구조", "4H/1H가 LH/LL 또는 주요 저항 아래", "상위 상승 추세의 정상 눌림"),
                ("저항 테스트", "이전 고점/박스 상단을 넘긴 뒤 종가가 안쪽으로 복귀", "상단 위에서 여러 종가가 유지되고 거래가 계속 붙음"),
                ("하위 촉발", "5m/15m 리테스트 실패, 낮은 고점 형성", "꼬리만 있고 종가 이탈이 없음"),
                ("무효화", "실패 돌파의 고점과 변동성 여유 위", "손절이 바로 상위 유동성 속에 놓임"),
                ("목표", "레인지 중앙은 보수적 TP1, 반대 하단은 다음 목표", "아래 지지가 바로 가까워 손익비가 부족"),
            ),
            styles,
            (28 * mm, 71 * mm, 73 * mm),
        ),
        PageBreak(),
    ]

    story += section("7. 진입 유형 C: 돌파가 아니라 '돌파 후 유지'를 거래한다", styles)
    story += [
        body(
            "돌파 캔들을 보고 추격하는 것은 신호를 늦게 읽는 가장 흔한 방식이다. 유효한 돌파의 핵심은 가격이 저항을 한 번 넘었는지가 아니라, "
            "<b>돌파 수준 위에서 종가를 유지하거나, 재시험 뒤 다시 그 수준을 지지로 바꾸는지</b>다. 이 과정은 실패 돌파를 줄이지만 "
            "진입 가격을 일부 포기한다.",
            styles,
        ),
        ChartScenario(
            "교육용 도식 - 종가 돌파와 리테스트", "breakout",
            "돌파 캔들 자체보다, 돌파 수준으로 돌아왔을 때 매수/매도가 어느 쪽을 지키는지를 본다.",
        ),
        Spacer(1, 3 * mm),
        callout(
            "<b>돌파 롱의 최소 문장:</b> '1H 박스 상단을 15m 종가로 넘겼다. 다음 15m가 상단 아래로 다시 마감하지 않았고, 5m "
            "리테스트도 상단을 지지했다. 리테스트 저점 아래면 돌파 가설은 무효다. 다음 상위 시간봉 유동성까지의 거리가 최소 목표다.'",
            styles,
        ),
        Spacer(1, 4 * mm),
        rule_table(
            ("상황", "진입 해석", "나갈 이유"),
            (
                ("종가만 돌파", "관찰 단계. 바로 추격하지 않음.", "다음 캔들이 박스 안으로 복귀"),
                ("돌파 뒤 얕은 리테스트", "지지 전환을 확인한 뒤 진입 후보", "리테스트 저점 이탈 또는 종가가 레벨 아래로 고정"),
                ("돌파 뒤 깊은 재진입", "돌파 실패 가능성. 상위 구조를 다시 확인", "손절을 넓혀서 원래 가설을 억지로 유지"),
            ),
            styles,
            (39 * mm, 65 * mm, 68 * mm),
        ),
        PageBreak(),
    ]

    story += section("8. 진입 유형 D: 레인지 가장자리 반전", styles)
    story += [
        body(
            "레인지에서는 가운데가 가장 위험하다. 위와 아래 어느 쪽에도 목표가 가깝고 무효화는 멀기 때문이다. 반전 거래는 상단이나 하단에서만 "
            "검토하고, <b>스윕 후 박스 안 복귀</b> 또는 <b>가장자리에서의 명확한 거절</b>이 있어야 한다.",
            styles,
        ),
        ChartScenario(
            "교육용 도식 - 레인지 하단 반전", "range",
            "중앙선 부근은 가격 정보가 적다. 가장자리에서만 가설과 무효화가 가까워진다.",
        ),
        Spacer(1, 3 * mm),
        rule_table(
            ("가격 위치", "검토", "TP 설계", "취소"),
            (
                ("하단 근처", "하단 스윕 뒤 15m 종가 회복이면 롱 후보", "TP1은 중앙선 또는 첫 저항, TP2는 상단", "15m 종가가 하단 밖에 유지"),
                ("상단 근처", "상단 스윕 뒤 종가 재진입이면 숏 후보", "TP1은 중앙선 또는 첫 지지, TP2는 하단", "15m 종가가 상단 밖에 유지"),
                ("중앙", "기다림", "손익비가 나쁘므로 목표를 억지로 멀리 잡지 않음", "중앙에서 새 포지션을 만들지 않음"),
            ),
            styles,
            (28 * mm, 55 * mm, 56 * mm, 33 * mm),
        ),
        PageBreak(),
    ]

    story += section("9. '촉발'은 꼬리가 아니라 종가와 재시험이다", styles)
    story += [
        body(
            "알트코인에서는 한 번의 긴 꼬리가 지지·저항을 넘나들 수 있다. 꼬리는 유동성 접근을 알려 주지만, 그 자체로 방향을 확정하지 않는다. "
            "따라서 촉발 규칙을 미리 두 가지 중 하나로 정한다: <b>종가 회복/이탈</b> 또는 <b>회복 뒤 리테스트 유지/실패</b>. "
            "둘을 섞어 쓰면 손절이 나간 뒤에만 규칙을 바꾸게 된다.",
            styles,
        ),
        rule_table(
            ("증거", "의미", "단독 사용 위험"),
            (
                ("긴 스윕 꼬리", "그 레벨에 주문과 유동성이 있었음", "그 뒤에도 같은 방향으로 계속 갈 수 있음"),
                ("시간봉 종가 회복", "레벨 밖 체류를 한 번 부정", "상위 시간봉과 충돌하면 반등일 수 있음"),
                ("리테스트 유지", "이전 저항/지지가 역할을 바꿀 가능성", "거래량·변동성 환경에 따라 미체결 또는 슬리피지 발생"),
                ("상대 거래량 증가", "관심·참여 증가 가능성", "단독 방향 신호가 아니며 뉴스성 급등락일 수 있음"),
            ),
            styles,
            (34 * mm, 68 * mm, 70 * mm),
        ),
        Spacer(1, 4 * mm),
        callout(
            "<b>체크 문장:</b> '꼬리가 나왔으니 들어간다'가 아니라 '꼬리 뒤 15m 종가가 구간을 회복했고, 다음 5m 리테스트에서 그 구간 "
            "아래로 다시 밀리지 않았다. 이 리테스트 저점이 깨지면 내가 본 회복은 실패다.'라고 적는다.",
            styles,
            Color(0.95, 0.98, 0.99),
        ),
        PageBreak(),
    ]

    story += section("10. SL은 '손실 허용액'이 아니라 가설의 무효화 가격", styles)
    story += [
        body(
            "손절을 진입가에서 고정 퍼센트로 떼면 시장 구조와 무관해진다. 먼저 <b>어느 가격에서 내 가설이 틀리는지</b>를 정하고, 그 뒤에 "
            "현재 변동성이 그 구조를 평범하게 흔들 수 있는지 확인한다. 구조적 손절이 너무 멀면 수량을 줄이거나 거래를 포기한다. 손절을 좁혀서 "
            "거래를 성립시키지 않는다.",
            styles,
        ),
        rule_table(
            ("구성", "방법", "주의"),
            (
                ("구조 기준", "롱은 회복 저점/지지 아래, 숏은 실패 돌파 고점/저항 위", "단순한 최근 캔들 저점이 아니라 설정의 논리를 깨는 지점"),
                ("변동성 여유", "현재 15m 또는 1H ATR%를 보고 구조 바로 밖에 작은 여유를 둠", "ATR은 방향을 예측하지 않으며, 큰 뉴스성 캔들에서는 평소 여유가 무의미할 수 있음"),
                ("시간 무효화", "정해 둔 시간 안에 촉발 뒤 기대 방향 진행이 없으면 재평가", "가격 SL만 있고 시간 SL이 없으면 횡보 포지션에 묶일 수 있음"),
                ("실행 기준", "종가 기준 무효화인지, 가격 도달 즉시 무효화인지 미리 고정", "사후에 '꼬리라서 무시'와 '종가라서 유지'를 번갈아 쓰지 않음"),
            ),
            styles,
            (27 * mm, 73 * mm, 72 * mm),
        ),
        Spacer(1, 4 * mm),
        body(
            "<b>롱의 계산 표현:</b> 위험 폭 R = 진입가 - SL. <b>숏:</b> R = SL - 진입가. R은 항상 양수여야 한다. "
            "손절 폭 비율 = R / 진입가. 이 비율이 현재 ATR%보다 지나치게 좁으면 정상 잡음에 닿기 쉽고, 지나치게 넓으면 목표와 손익비가 "
            "나빠질 수 있다. ATR%는 종목·시간봉마다 따로 보아야 하며, 서로 다른 코인의 절대 ATR을 비교하면 안 된다.",
            styles,
        ),
        PageBreak(),
    ]

    story += section("11. TP는 '희망 가격'이 아니라 반대 유동성과 R의 교차점", styles)
    story += [
        body(
            "목표가는 진입 뒤에 정하는 숫자가 아니다. 진입 전에 반대 주문이 쌓일 가능성이 있는 구간을 표시하고, 그 거리가 감수하는 R에 비해 충분한지 본다. "
            "TP를 여러 개로 나누는 목적은 방향을 더 잘 맞히는 것이 아니라, 목표마다 포지션의 성격을 바꾸기 위해서다.",
            styles,
        ),
        rule_table(
            ("목표", "차트 근거", "포지션 관리 의미"),
            (
                ("TP1", "가장 가까운 반대 구조: 직전 고점/저점, 레인지 중앙, 첫 공급·수요 구간", "일부 실현 후 남은 포지션의 리스크를 다시 정의"),
                ("TP2", "상위 시간봉 반대 유동성 또는 다음 박스 경계", "초기 가설이 여전히 유효한지 확인하며 보유"),
                ("잔여분", "추세 확장이 실제로 유지될 때만, 후행 구조 아래/위로 관리", "TP1을 맞았다고 원래 SL을 무조건 진입가로 옮기지 않음"),
                ("전량 퇴장", "구조 반대 종가, 촉발 실패, 시간 무효화, 목표 도달", "기대가 바뀌었으면 새 거래로 다시 판단"),
            ),
            styles,
            (22 * mm, 82 * mm, 68 * mm),
        ),
        Spacer(1, 4 * mm),
        callout(
            "<b>R 배수로 검산:</b> 롱에서 TP1 = 진입가 + 1R, TP2 = 진입가 + 2R처럼 R 배수는 통일된 언어가 된다. 다만 차트상 첫 저항이 "
            "0.6R에 있으면 '2R 목표'라는 이유로 그 저항을 무시하지 않는다. 실제 구조와 R 배수 중 먼저 오는 곳이 보수적 첫 목표다.",
            styles,
            Color(1.0, 0.975, 0.94),
        ),
        PageBreak(),
    ]

    story += section("12. 레버리지는 신호의 품질을 높이지 않는다", styles)
    story += [
        body(
            "레버리지 설정과 포지션 크기는 별개다. 차트가 먼저 SL을 결정하고, 계좌에서 한 번의 가설에 감수할 금액이 그 다음에 수량을 결정한다. "
            "원하는 레버리지 숫자에 맞추기 위해 수량을 키우면, 구조가 아니라 청산 가격이 거래를 지배하게 된다.",
            styles,
        ),
        rule_table(
            ("계산", "의미", "예시 - 교육용"),
            (
                ("위험금액", "본인이 사전에 정한 한 거래 최대 허용 손실", "5 USDT를 넘기지 않기로 정함"),
                ("손절 폭", "|진입가 - SL| / 진입가", "진입 100, SL 98이면 2%"),
                ("명목 포지션", "위험금액 / 손절 폭", "5 / 0.02 = 250 USDT"),
                ("실효 레버리지", "명목 포지션 / 계좌 순자산", "순자산 1,000이면 0.25배 노출"),
            ),
            styles,
            (34 * mm, 65 * mm, 73 * mm),
        ),
        Spacer(1, 4 * mm),
        body(
            "선물에서는 거래소의 마크 가격, 유지증거금, 수수료, 펀딩, 마진 모드, 종목별 티어가 실제 청산 위험에 영향을 준다. "
            "Bitget은 청산 위험을 마크 가격과 증거금 요건으로 평가한다고 설명한다. 따라서 차트의 마지막 거래가만 보고 'SL이 청산보다 앞에 있다'고 "
            "단정할 수 없다. <link href='https://www.bitget.com/support/articles/360058821652' color='blue'>Bitget 레버리지 및 리스크 관리 문서</link>를 주문 직전에 다시 확인한다.",
            styles,
        ),
        PageBreak(),
    ]

    story += section("13. 주문을 넣기 전 - 가격, 트리거, 체결은 다르다", styles)
    story += [
        body(
            "차트의 진입·SL·TP 선은 거래소에서 자동으로 같은 가격에 체결된다는 보장이 아니다. 트리거 가격은 주문을 시장에 보내는 조건이고, "
            "그 뒤 시장가/지정가 여부와 유동성에 따라 실제 체결가가 달라질 수 있다. 특히 변동성이 큰 알트코인은 갭과 슬리피지를 계획에 포함해야 한다.",
            styles,
        ),
        rule_table(
            ("구분", "차트에서의 뜻", "실행 전에 확인할 것"),
            (
                ("진입가", "가설을 시작할 가격 또는 영역", "지정가인지, 확인 뒤 시장가인지, 미체결이면 취소하는지"),
                ("SL 트리거", "가설 무효화가 감지되는 가격", "Last / Mark / Index 중 무엇이 트리거인지"),
                ("SL 실행가", "트리거 뒤 실제 청산 주문이 나가는 방식", "시장가의 슬리피지, 지정가 미체결 위험"),
                ("TP", "반대 유동성에서 실현할 구간", "부분 청산 수량, reduce-only, 나머지 SL과 충돌 여부"),
            ),
            styles,
            (30 * mm, 68 * mm, 74 * mm),
        ),
        Spacer(1, 4 * mm),
        body(
            "Bitget의 안내에 따르면 선물 TP/SL은 Last, Mark, Index 가격을 트리거로 선택할 수 있고, 청산에는 마크 가격이 사용된다. "
            "또한 트리거 주문은 트리거에 닿은 뒤에야 시장으로 제출된다. 그러므로 차트의 한 선을 그대로 복사하기 전에 트리거 종류와 주문 종류를 "
            "한 번 더 읽어야 한다. <link href='https://www.bitget.com/asia/support/articles/12560603817159' color='blue'>TP/SL 트리거 가격 안내</link>, "
            "<link href='https://www.bitget.com/support/articles/12560603828238' color='blue'>트리거와 체결 불일치 안내</link>를 참고한다.",
            styles,
        ),
        PageBreak(),
    ]

    story += section("14. 진입 뒤의 관리: 맞았을 때보다 틀렸을 때의 규칙", styles)
    story += [
        body(
            "포지션 관리 규칙은 이익일 때의 탐욕보다 손실일 때의 즉흥성을 줄이기 위해 존재한다. TP1 뒤 무조건 본절로 올리는 방식도, "
            "손절을 넓혀 버티는 방식도 모든 상태에서 옳지 않다. 처음 세운 구조 기준이 바뀌었을 때만 계획을 바꾼다.",
            styles,
        ),
        rule_table(
            ("사건", "기계적으로 확인할 것", "가능한 행동"),
            (
                ("진입 뒤 바로 역행", "촉발 저점/고점이 유지되는가", "무효화면 SL. 재진입은 별도 새 가설로만."),
                ("TP1 도달", "상위 구조와 다음 목표까지의 경로가 아직 유효한가", "부분 실현 후 잔여분의 SL을 새 구조 아래/위로 재정의"),
                ("목표 앞 강한 반대 종가", "목표 유동성이 흡수됐는지, 반전 촉발이 생겼는지", "일부 또는 전량 청산을 계획 규칙으로 실행"),
                ("시간만 경과", "설정이 기대한 시간 안에 진행됐는가", "시간 무효화: 축소·퇴장·다시 관찰"),
                ("추가 진입 욕구", "원래 손절과 평균 단가가 바뀌는가", "처음 계획에 없었다면 추가 진입하지 않음"),
            ),
            styles,
            (33 * mm, 75 * mm, 64 * mm),
        ),
        Spacer(1, 4 * mm),
        callout(
            "<b>중요:</b> 한 번 손절 난 설정은 '원래 방향은 맞았다'고 해도 실패한 거래다. 손절 뒤 반대 방향으로 움직였는지와 별개로, "
            "해당 진입·무효화·체결 규칙이 기대값을 가졌는지는 기록을 여러 번 쌓은 뒤에만 판단할 수 있다.",
            styles,
        ),
        PageBreak(),
    ]

    story += section("15. 6분 차트 루틴 - 실제로 손으로 하는 순서", styles)
    story += [
        rule_table(
            ("시간", "행동", "기록할 한 줄"),
            (
                ("0:00 - 1:00", "4H/1H에서 최근 두 고점·저점과 큰 박스 상·하단 표시", "상위 상태: 상승 / 하락 / 레인지"),
                ("1:00 - 2:00", "현재 가격이 표시한 구간의 끝인지, 가운데인지 확인", "위치: 지지 / 저항 / 중앙 / 이미 목표 근처"),
                ("2:00 - 3:00", "15m ATR%와 최근 급등락 여부 확인", "변동성: 정상 / 확장 / 관망"),
                ("3:00 - 4:00", "15m 촉발 조건을 하나만 선택: 종가 회복 또는 리테스트", "촉발: 아직 없음 / 관찰 중 / 확인"),
                ("4:00 - 5:00", "진입 영역, 구조 SL, TP1/TP2를 선으로 표시", "R, 목표 경로, 시간 무효화"),
                ("5:00 - 6:00", "수량·트리거 타입·주문 방식·취소 조건 확인", "들어가도 되는가? 아니면 알림만 둘 것인가?"),
            ),
            styles,
            (26 * mm, 96 * mm, 50 * mm),
        ),
        Spacer(1, 5 * mm),
        callout(
            "<b>관망 판정 예:</b> '4H는 상승이지만 가격이 1H 박스 중앙에 있고, 15m 촉발도 없으며 첫 저항까지 0.5R밖에 남지 않았다. "
            "따라서 롱 신호가 아니라 기다림.' 이렇게 쓰면 FOMO와 사후합리화를 줄일 수 있다.",
            styles,
            Color(0.98, 0.98, 0.94),
        ),
        PageBreak(),
    ]

    story += section("16. 신호를 검증할 때: 결과를 보지 않는 '시점 고정' 재생", styles)
    story += [
        body(
            "텔레그램의 특정 타점을 분석하려면, 사후 차트로 '맞았네'라고 보지 않는다. 게시 시각 바로 전까지의 캔들만 보이게 고정하고, "
            "그 시점에 실제로 주문을 낼 수 있었는지를 따로 판정한다. 이것이 신호의 원리와 결과 운을 분리하는 가장 중요한 절차다.",
            styles,
        ),
        rule_table(
            ("기록 항목", "왜 필요한가", "판정 예"),
            (
                ("원문과 게시 시각", "어떤 캔들까지 알고 있었는지 고정", "메시지 수정·재전송·시간대도 원문 그대로 기록"),
                ("종목/계약과 차트 가격", "현물·선물·거래소 가격 차이를 구분", "USDT 무기한인지, 어떤 거래소의 가격인지"),
                ("진입 영역과 체결", "가격이 닿았는지와 실제 체결 가능성은 다름", "지정가가 도달 전인지, 스파이크만 닿았는지"),
                ("SL/TP 순서", "한 봉 안에서 둘 다 닿으면 결과를 단정할 수 없음", "더 낮은 시간봉 또는 보수적 체결 가정으로 표시"),
                ("시장 상태", "신호가 어떤 환경에서만 통하는지 발견", "추세 눌림, 레인지 끝, 뉴스성 확장 등"),
            ),
            styles,
            (34 * mm, 73 * mm, 65 * mm),
        ),
        Spacer(1, 4 * mm),
        callout(
            "<b>평가 순서:</b> ① 시간 고정 → ② 그 시점의 구조 문장 작성 → ③ 실제 진입 가능 여부 → ④ SL/TP 순서와 비용 → "
            "⑤ 동일 조건의 여러 신호를 묶어 비교. 단일 성공 사례나 채널의 사후 캡처는 원리의 증거가 아니다.",
            styles,
        ),
        PageBreak(),
    ]

    story += section("17. 직접 작성하는 한 장짜리 트레이드 시트", styles)
    story += [
        body("아래 표를 한 거래마다 채우면, 신호를 숫자에서 구조로 바꿀 수 있다. 빈 칸이 남으면 주문도 비워 둔다.", styles),
        rule_table(
            ("필드", "내가 적을 내용", "예시 형식"),
            (
                ("시간 고정", "게시/관찰 시각, 시간대, 마지막 완료 캔들", "2026-09-04 14:15 KST, 15m 캔들 마감"),
                ("종목과 시장", "계약, 거래소, 현물/무기한, 가격 기준", "SOLUSDT 무기한, Bitget, mark/last 확인"),
                ("상위 구조", "4H/1H 방향과 핵심 레벨", "4H HH/HL, 1H 지지 재시험"),
                ("설정 유형", "눌림 / 돌파 리테스트 / 레인지 반전 / 관망", "상승 추세 눌림 롱"),
                ("촉발", "무슨 종가·리테스트가 확인되어야 하나", "15m 지지 위 종가 + 5m 리테스트 유지"),
                ("진입 / 취소", "진입 영역과 체결 전 취소 조건", "영역 A-B, 15m 종가가 지지 아래면 취소"),
                ("SL / 시간 SL", "구조 무효화, 가격 또는 종가 기준, 시간", "회복 저점 아래 + 변동성 여유, 3개 15m봉"),
                ("TP1 / TP2", "반대 구조와 R 배수", "첫 저항 또는 1R / 4H 유동성 또는 2R"),
                ("수량", "위험금액에서 역산한 명목 포지션", "위험금액 ÷ 손절 폭"),
                ("사후 기록", "체결, 비용, 계획 준수, 배운 점", "방향 아닌 규칙 준수 여부를 분리"),
            ),
            styles,
            (29 * mm, 81 * mm, 62 * mm),
        ),
        PageBreak(),
    ]

    story += section("18. 즉시 보류해야 하는 경고 신호", styles)
    story += [
        rule_table(
            ("경고", "왜 위험한가", "대신 할 일"),
            (
                ("이미 큰 캔들이 나온 뒤 신호를 봄", "진입과 무효화의 거리가 바뀌어 원래 손익비가 사라짐", "원래 진입 영역으로 돌아오는지 기다리거나 취소"),
                ("차트 중앙에서 TP가 멂", "가까운 반대 구조를 무시한 목표가 됨", "가장자리까지 기다리거나 관망"),
                ("SL이 청산 예상가와 가까움", "마크 가격·유지증거금·비용 때문에 계획대로 작동하지 않을 수 있음", "포지션 크기 축소 또는 거래 포기, 거래소 계산기 재확인"),
                ("뉴스성 수직 캔들", "평상시 ATR 기반 SL/TP의 전제가 무너짐", "변동성 안정 전까지 관망"),
                ("원문이 '확실', '무조건'만 말함", "무효화와 체결 조건이 없는 신호", "진입·SL·TP·취소 조건을 먼저 스스로 작성"),
                ("손절 뒤 즉시 복수 진입", "새 근거 없이 감정만 바뀐 상태", "최소 한 개 상위/중간 시간봉 종가 후 재평가"),
            ),
            styles,
            (40 * mm, 72 * mm, 60 * mm),
        ),
        Spacer(1, 5 * mm),
        callout(
            "<b>마지막 게이트:</b> '손절이 나가도 내가 왜 나갔는지 한 문장으로 설명할 수 있는가?'와 'TP1이 닿기 전에 손절이 날 "
            "확률을 줄여 주는 구조 근거가 있는가?' 두 질문 모두에 답하지 못하면 거래가 아니라 추측이다.",
            styles,
            Color(1.0, 0.955, 0.955),
        ),
        PageBreak(),
    ]

    story += section("19. 거래소 실행 전 최소 점검", styles)
    story += [
        body(
            "차트 해석이 맞아도 주문 설정이 다르면 다른 거래가 된다. 아래 항목은 전략이 아니라 실행 오류를 줄이기 위한 점검표다. "
            "거래소 UI와 계약 규칙은 바뀔 수 있으므로, 실제 주문 전 현재 화면과 공식 도움말을 확인한다.",
            styles,
        ),
        rule_table(
            ("점검", "확인할 내용"),
            (
                ("계약", "종목명, USDT-M/COIN-M, 현물/선물, 교차/격리 마진"),
                ("가격", "차트 가격이 Last인지 Mark인지, TP/SL 트리거 타입이 무엇인지"),
                ("수량", "계획 위험금액에서 역산했는지, 레버리지 숫자를 채우려고 키우지 않았는지"),
                ("SL", "무효화 조건, 가격/종가 기준, 트리거·실행 방식, 슬리피지 가정"),
                ("TP", "부분 청산 수량, reduce-only, 잔여 수량의 SL, TP끼리 충돌하지 않는지"),
                ("청산", "거래소 계산기·유지증거금·마크 가격 기반 위험을 현재 값으로 확인했는지"),
                ("취소", "체결 전 무엇이 일어나면 주문을 취소할지, 시간 무효화가 있는지"),
            ),
            styles,
            (36 * mm, 136 * mm),
        ),
        Spacer(1, 4 * mm),
        body(
            "Bitget은 TP/SL을 가격, ROI, 가격 변동률, PnL 등의 방식으로 설정할 수 있다고 안내한다. 이 자료의 훈련 순서는 <b>가격 구조 → "
            "가격 기반 SL/TP → 수량</b>이다. ROI나 PnL 입력은 최종 표시 형식일 수 있지만, 차트 가설 자체를 대신하지는 않는다. "
            "<link href='https://www.bitget.com/support/articles/12560603820669' color='blue'>Bitget TP/SL 설정 안내</link>를 참조한다.",
            styles,
        ),
        PageBreak(),
    ]

    story += section("20. 핵심 요약", styles)
    story += [
        callout(
            "<b>차트로 타점을 고르는 순서:</b> 상위 시간봉에서 거래할 구간을 고른다 → 중간 시간봉에서 설정 유형을 분류한다 → "
            "하위 시간봉에서 종가/리테스트 촉발을 기다린다 → 구조가 깨지는 가격을 SL로 둔다 → 반대 유동성과 R을 교차해 TP를 둔다 → "
            "위험금액으로 수량을 역산한다 → 촉발 전/후 취소 조건과 시간 무효화를 기록한다.",
            styles,
            Color(0.92, 0.97, 0.96),
        ),
        Spacer(1, 5 * mm),
        body(
            "이 과정을 거치면 텔레그램 신호를 '롱/숏, 진입가, TP, SL'의 네 숫자로 보지 않게 된다. 대신 '어떤 상태에서, 어느 구간에서, "
            "무슨 증거가 나오면, 어디에서 틀렸다고 인정하고, 어디에서 이익을 실현하는가'라는 재현 가능한 가설로 바뀐다. "
            "그 가설이 좋은지는 단일 차트가 아니라 시간 고정된 여러 사례의 체결·비용·실패까지 기록해 검증해야 한다.",
            styles,
        ),
        Spacer(1, 5 * mm),
        rule_table(
            ("이 문서로 할 수 있는 것", "이 문서가 보장하지 않는 것"),
            (
                ("신호를 구조·무효화·목표로 분해하고, 조건이 없으면 관망하는 연습", "특정 코인의 다음 가격, 수익, 청산 회피, 특정 텔레그램 채널의 적중률"),
                ("원문과 시각을 받았을 때 편향 없이 사례를 재생·검증하는 기록 틀", "과거 사례 하나로 미래 매매 결정을 정당화하는 일"),
            ),
            styles,
            (85 * mm, 87 * mm),
        ),
        Spacer(1, 7 * mm),
        p("공식 자료와 참고", styles["h2"]),
        body(
            "• <link href='https://www.tradingview.com/support/solutions/43000501823-average-true-range-atr/' color='blue'>TradingView - Average True Range (ATR)</link>: ATR의 정의와 방향 예측 한계.<br/>"
            "• <link href='https://www.tradingview.com/support/solutions/43000695003-average-daily-range-adr-indicator/' color='blue'>TradingView - Average Daily Range (ADR)</link>: 변동성 범위와 목표/손절의 현실성 점검.<br/>"
            "• <link href='https://www.bitget.com/support/articles/360058821652' color='blue'>Bitget - Leverage and Risk Management</link>: 마크 가격, 유지증거금, 청산 리스크.<br/>"
            "• <link href='https://www.bitget.com/asia/support/articles/12560603817159' color='blue'>Bitget - TP/SL trigger price options</link>: Last/Mark/Index 트리거 차이.<br/>"
            "• <link href='https://www.bitget.com/support/articles/12560603820669' color='blue'>Bitget - How to Set TP/SL in Futures Trading</link>: TP/SL 설정 방식.",
            styles,
        ),
    ]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    print(build())
