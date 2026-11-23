"""The main window: header with issue time and input health, the map with lead-time and history sliders, and the
commune summary, commune table, bulletin draft and product notes on the right."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QGuiApplication,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..product import AreaWarning, Forecast, Level
from .bulletin import bulletin, geojson, inputs_line
from .geo import Geo
from .mapview import Base, Layers, MapCanvas
from .store import ForecastStore, Issue
from .style import (
    GOOD,
    INK,
    INK_2,
    LEVEL_COLOR,
    LEVEL_ICON,
    LEVEL_RANK,
    LEVEL_TEXT,
    MAP,
    MUTED,
    PANEL,
    RULE,
    SOURCE_NAMES,
    STALE_AFTER,
    STALE_COLOR,
    SURFACE,
    day,
    hhmm,
    ink_on,
    prob_color,
)

REFRESH_MS = 15_000
STYLE = f"""
QWidget {{ background: {SURFACE}; color: {INK}; font-size: 10pt; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {RULE}; border-radius: 6px; }}
QLabel#muted {{ color: {INK_2}; }}
QTabWidget::pane {{ border: 1px solid {RULE}; }}
QTableWidget {{ gridline-color: {RULE}; }}
QTableWidget::item:selected {{ background: {RULE}; color: {INK}; }}
QHeaderView::section {{ background: {PANEL}; color: {INK_2}; border: none; border-bottom: 1px solid {RULE}; padding: 4px; }}
QPushButton {{ background: {PANEL}; border: 1px solid {RULE}; border-radius: 4px; padding: 4px 10px; }}
QPushButton:hover {{ border-color: {INK_2}; }}
"""


CLOCK: datetime | None = None  # replays: the time the app pretends it is (UTC), set by --clock


def utcnow() -> datetime:
    return CLOCK if CLOCK is not None else datetime.now(UTC).replace(tzinfo=None)


def level_badge(level: Level) -> str:
    """Rich text: icon + label on the level's color (never color alone)."""
    color = LEVEL_COLOR[level]
    if color is None:
        return f"<span style='color:{INK_2}'>{LEVEL_ICON[level]} {LEVEL_TEXT[level]}</span>"
    return f"<span style='background:{color}; color:{ink_on(color)}; padding:2px 6px;'>&nbsp;{LEVEL_ICON[level]} {LEVEL_TEXT[level]}&nbsp;</span>"


def window_label(fc: Forecast, k: int) -> str:
    """Window k as its end time and radius, e.g. "18:00 (10 km)" (headers and marks)."""
    w = fc.windows[k]
    return f"{hhmm(fc.anchor + timedelta(minutes=w.end_min))} ({w.radius_km:g} km)"


def until(fc: Forecast, k: int) -> str:
    """Window k in a sentence, e.g. "đến 18:00 (10 km)"."""
    return f"đến {window_label(fc, k)}"


def duration(td: timedelta) -> str:
    m = int(td.total_seconds() // 60)
    return f"{m} phút" if m < 120 else f"{m // 60} giờ" if m < 48 * 60 else f"{m // 1440} ngày"


def change_text(now: AreaWarning, before: AreaWarning | None, k: int) -> str:
    was = before.level if before else Level.NONE
    if LEVEL_RANK[now.level] > LEVEL_RANK[was]:
        return "Mới" if was == Level.NONE else "▲ Nâng cấp"
    if LEVEL_RANK[now.level] < LEVEL_RANK[was]:
        return "Kết thúc" if now.level == Level.NONE else "▼ Hạ cấp"
    if before is not None and abs(now.probability[k] - before.probability[k]) >= 0.05:
        return f"{(now.probability[k] - before.probability[k]) * 100:+.0f}%"
    return ""


def province_short(province: str) -> str:
    """'Tỉnh Đồng Tháp' -> 'Đồng Tháp', 'Thành phố Hồ Chí Minh' -> 'TP. Hồ Chí Minh' (table and summary)."""
    for long, short in (("Tỉnh ", ""), ("Thành phố ", "TP. ")):
        if province.startswith(long):
            return short + province[len(long):]
    return province


def level_summary(fc: Forecast, prev: Forecast | None, level: Level) -> str:
    """Rich text: the communes at `level`, the provinces with the most of them, and the change since the previous forecast."""
    here = [a for a in fc.communes if a.level == level]
    by: dict[str, int] = {}
    for a in here:
        by[a.province] = by.get(a.province, 0) + 1
    top = ", ".join(f"{province_short(p)} {n}" for p, n in sorted(by.items(), key=lambda x: -x[1])[:4]) + (", …" if len(by) > 4 else "")
    text = level_badge(level) + f"&nbsp; <b>{len(here)}</b> phường/xã" + (f" &nbsp;<span style='color:{INK_2}'>{top}</span>" if by else "")
    if prev is not None:
        was = {a.code for a in prev.communes if a.level == level}
        new, gone = len({a.code for a in here} - was), len(was - {a.code for a in here})
        if new or gone:
            text += f" &nbsp;<span style='color:{INK_2}'>(so với bản trước: +{new} / −{gone})</span>"
    return text


class MainWindow(QMainWindow):
    def __init__(self, store: ForecastStore, geo: Geo, hours: float, start: datetime | None = None) -> None:
        super().__init__()
        self.store, self.geo, self.hours = store, geo, hours
        self.notable = {c.code for c in geo.communes if c.notable}
        self.provinces = list(dict.fromkeys(c.province for c in geo.communes))  # bulletin order
        self.issues: list[Issue] = []
        self.index = -1
        self.k = 0
        self.layers = Layers()
        self.fc: Forecast | None = None
        self.prev: Forecast | None = None
        self.setWindowTitle("Dự báo sét – Nam Bộ")
        self.setStyleSheet(STYLE)

        # header -------------------------------------------------------------------------------------------
        self.title = QLabel("DỰ BÁO SÉT 0–2 GIỜ · NAM BỘ")
        self.title.setFont(QFont(self.title.font().family(), 14, QFont.Weight.Bold))
        self.issue_info = QLabel()
        self.stale = QLabel()
        self.stale.setStyleSheet(f"background:{STALE_COLOR}; color:white; padding:4px 8px; border-radius:4px; font-weight:bold;")
        self.stale.hide()
        self.health = QLabel()
        self.issue_info.setWordWrap(True)
        self.lead_text = QLabel()
        header = QHBoxLayout()
        header.addWidget(self.title)
        header.addSpacing(16)
        header.addWidget(self.issue_info, 1)
        header.addWidget(self.stale)
        header.addSpacing(12)
        header.addWidget(self.health)

        # map column ---------------------------------------------------------------------------------------
        self.canvas = MapCanvas(geo)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.hover = QLabel(" ")
        self.hover.setObjectName("muted")
        self.verify_text = QLabel()
        self.canvas.hovered.connect(self.hover.setText)

        self.base = QComboBox()
        self.base.addItems([b.value for b in Base])
        self.base.currentIndexChanged.connect(lambda i: self._set_layer("base", list(Base)[i]))
        toggles = QGridLayout()
        toggles.addWidget(QLabel("Lớp nền:"), 0, 0)
        toggles.addWidget(self.base, 1, 0)
        for i, (attr, text) in enumerate((("contours", "Đường 30/40%"), ("strokes", "Sét 60 phút qua"), ("radar_contour", "Radar 35 dBZ"),
                                          ("communes", "Phường/xã theo cấp"), ("verify", "Sét trong khung (kiểm chứng)"),
                                          ("rings", "Vòng 120/300 km"))):
            box = QCheckBox(text)
            box.setChecked(getattr(self.layers, attr))
            box.toggled.connect(lambda on, a=attr: self._set_layer(a, on))
            toggles.addWidget(box, i // 3, 1 + i % 3)
        toggles.setColumnStretch(4, 1)
        reset = QPushButton("Toàn vùng")
        reset.clicked.connect(self.canvas.reset_view)
        toggles.addWidget(reset, 0, 5)

        self.lead = QSlider(Qt.Orientation.Horizontal)
        self.lead.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.lead.setPageStep(1)
        self.lead.valueChanged.connect(self._set_window)
        self.lead_marks = QHBoxLayout()
        self.marks: list[QLabel] = []
        self.lead_text.setFont(QFont(self.lead_text.font().family(), 11, QFont.Weight.Bold))
        self.play = QPushButton("▶ Chạy")
        self.play.setCheckable(True)
        self.play.toggled.connect(lambda on: self.play_timer.start(1200) if on else self.play_timer.stop())
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(lambda: self.lead.setValue((self.lead.value() + 1) % (self.lead.maximum() + 1)))
        lead_box = QGridLayout()
        lead_box.addWidget(QLabel("Hạn dự báo:"), 0, 0)
        lead_box.addWidget(self.lead, 0, 1)
        lead_box.addWidget(self.play, 0, 2)
        lead_box.addLayout(self.lead_marks, 1, 1)
        lead_box.addWidget(self.lead_text, 2, 1)
        lead_box.addWidget(self.verify_text, 3, 1)

        self.history = QSlider(Qt.Orientation.Horizontal)
        self.history.setPageStep(6)
        self.history.valueChanged.connect(self._set_issue)
        back, forward = QPushButton("◀"), QPushButton("▶")
        back.clicked.connect(lambda: self.history.setValue(self.history.value() - 1))
        forward.clicked.connect(lambda: self.history.setValue(self.history.value() + 1))
        newest = QPushButton("Mới nhất")
        newest.clicked.connect(lambda: self.history.setValue(self.history.maximum()))
        self.follow = QCheckBox("Tự động theo bản mới nhất")
        self.follow.setChecked(True)
        self.history_text = QLabel()
        hist_box = QHBoxLayout()
        hist_box.addWidget(QLabel("Bản tin:"))
        hist_box.addWidget(back)
        hist_box.addWidget(self.history, 1)
        hist_box.addWidget(forward)
        hist_box.addWidget(self.history_text)
        hist_box.addWidget(newest)
        hist_box.addWidget(self.follow)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.canvas, 1)
        lv.addWidget(self.hover)
        lv.addLayout(toggles)
        lv.addLayout(lead_box)
        lv.addLayout(hist_box)

        # right column -------------------------------------------------------------------------------------
        self.summary = {lv: QLabel() for lv in (Level.WARNING, Level.ALERT)}
        card = QFrame()
        card.setObjectName("card")
        cv = QVBoxLayout(card)
        cv.setContentsMargins(10, 8, 10, 8)
        for lab in self.summary.values():
            lab.setTextFormat(Qt.TextFormat.RichText)
            lab.setWordWrap(True)
            cv.addWidget(lab)
        self.province_filter = QComboBox()
        self.province_filter.addItems(["Tất cả tỉnh/thành"] + [province_short(p) for p in self.provinces])
        self.province_filter.currentIndexChanged.connect(lambda _: self._fill_table())
        self.only_levels = QCheckBox("Chỉ phường/xã theo dõi / cảnh báo")
        self.only_levels.setChecked(True)
        self.only_levels.toggled.connect(lambda _: self._fill_table())
        self.only_notable = QCheckBox("Chỉ phường/xã tiêu biểu")
        self.only_notable.toggled.connect(lambda _: self._fill_table())
        self.table_count = QLabel()
        self.table_count.setObjectName("muted")
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().hide()
        self.table.itemSelectionChanged.connect(self._on_row)
        filt = QHBoxLayout()
        filt.addWidget(self.province_filter)
        filt.addWidget(self.only_levels)
        filt.addWidget(self.only_notable)
        filt.addStretch(1)
        filt.addWidget(self.table_count)
        dist_tab = QWidget()
        dv = QVBoxLayout(dist_tab)
        dv.addLayout(filt)
        dv.addWidget(self.table, 1)

        self.text = QPlainTextEdit()
        buttons = QHBoxLayout()
        for label, slot in (("Tạo lại", self._fill_bulletin), ("Sao chép", self._copy), ("Lưu văn bản…", self._save_text),
                            ("Xuất GeoJSON…", self._save_geojson)):
            b = QPushButton(label)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        buttons.addStretch(1)
        bull_tab = QWidget()
        bv = QVBoxLayout(bull_tab)
        bv.addWidget(QLabel("Bản tin dự thảo – dự báo viên kiểm tra và chỉnh sửa trước khi phát hành."))
        bv.addWidget(self.text, 1)
        bv.addLayout(buttons)

        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setAlignment(Qt.AlignmentFlag.AlignTop)
        info_tab = QWidget()
        QVBoxLayout(info_tab).addWidget(self.info)

        tabs = QTabWidget()
        tabs.addTab(dist_tab, "Phường/xã")
        tabs.addTab(bull_tab, "Bản tin")
        tabs.addTab(info_tab, "Thông tin")
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(card)
        rv.addWidget(tabs, 1)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([830, 780])  # the table needs ~770 px for commune names and five windows
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.addLayout(header)
        outer.addWidget(split, 1)
        self.setCentralWidget(root)

        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self.history.setValue(self.history.value() - 1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self.history.setValue(self.history.value() + 1))
        QShortcut(QKeySequence(Qt.Key.Key_Up), self, lambda: self.lead.setValue(self.lead.value() + 1))
        QShortcut(QKeySequence(Qt.Key.Key_Down), self, lambda: self.lead.setValue(self.lead.value() - 1))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(REFRESH_MS)
        self.refresh()
        if start is not None and (i := next((i for i, x in enumerate(self.issues) if x.anchor == start), None)) is not None:
            self.follow.setChecked(False)
            self.history.setValue(i)

    # ------------------------------------------------------------------------------------------- data flow
    def refresh(self) -> None:
        """Pick up new forecasts; follow the newest if asked."""
        issues = self.store.issues(self.hours)
        if issues and (not self.issues or issues[-1].anchor != self.issues[-1].anchor or len(issues) != len(self.issues)):
            current = self.issues[self.index].anchor if 0 <= self.index < len(self.issues) else None
            at_end = self.index == len(self.issues) - 1
            self.issues = issues
            self.history.blockSignals(True)
            self.history.setRange(0, len(issues) - 1)
            self.history.blockSignals(False)
            if self.follow.isChecked() or at_end or current is None:
                target = len(issues) - 1
            else:
                target = next((i for i, x in enumerate(issues) if x.anchor == current), len(issues) - 1)
            if target == self.history.value():
                self._set_issue(target)
            else:
                self.history.setValue(target)
        self._update_clock()

    def _set_issue(self, i: int) -> None:
        if not self.issues:
            return
        self.index = max(0, min(i, len(self.issues) - 1))
        issue = self.issues[self.index]
        self.fc = self.store.load(issue)
        self.prev = self.store.previous(issue.anchor)
        self._build_lead_marks(self.fc)
        self.history_text.setText(f"{hhmm(self.fc.issued)} {day(self.fc.issued)}")
        self._render()
        self._fill_bulletin()

    def _set_window(self, k: int) -> None:
        self.k = k
        if self.fc is not None:
            self._render()

    def _set_layer(self, attr: str, value: object) -> None:
        setattr(self.layers, attr, value)
        if self.fc is not None:
            self._render(map_only=True)

    def _build_lead_marks(self, fc: Forecast) -> None:
        if self.lead.maximum() != len(fc.windows) - 1:  # first forecast: start on the levels' window
            self.lead.blockSignals(True)
            self.lead.setRange(0, len(fc.windows) - 1)
            self.lead.setValue(fc.level_index)
            self.lead.blockSignals(False)
            self.k = self.lead.value()
        while len(self.marks) < len(fc.windows):
            mark = QLabel()
            mark.setObjectName("muted")
            self.marks.append(mark)
            self.lead_marks.addWidget(mark, 1)
        for i, mark in enumerate(self.marks):
            last = len(fc.windows) - 1
            mark.setAlignment(Qt.AlignmentFlag.AlignLeft if i == 0 else Qt.AlignmentFlag.AlignRight if i == last else Qt.AlignmentFlag.AlignCenter)
            mark.setText(window_label(fc, i) if i <= last else "")

    # ------------------------------------------------------------------------------------------- views
    def _render(self, map_only: bool = False) -> None:
        fc, k = self.fc, self.k
        assert fc is not None
        w = fc.windows[k]
        start, end = fc.anchor + timedelta(minutes=10), fc.anchor + timedelta(minutes=w.end_min)
        self.lead_text.setText(f"Có sét trong bán kính {w.radius_km:g} km từ {hhmm(start)} đến {hhmm(end)}"
                               + ("  (mức theo dõi / cảnh báo tính trên khung này)" if k == fc.level_index else ""))
        verification = self.store.verification(fc.anchor, w.end_min) if self.index < len(self.issues) - 1 or end < utcnow() else None
        self.canvas.show(fc, k, self.layers, verification)
        self.verify_text.setText("" if verification is None else
                                 f"<span style='color:{MAP.verify}'>✕</span> Kiểm chứng: {verification.strokes.time.size} cú sét trong khung "
                                 f"(số liệu đến {hhmm(verification.until)})")
        if map_only:
            return
        for lv, lab in self.summary.items():
            lab.setText(level_summary(fc, self.prev, lv))
        self._fill_table()
        self._fill_info()
        self._update_clock()

    def _update_clock(self) -> None:
        if self.fc is None:
            self.issue_info.setText("Chưa có bản tin trong thư mục làm việc.")
            return
        fc, now = self.fc, utcnow()
        self.issue_info.setText(f"Bản tin {hhmm(fc.issued)} ngày {day(fc.issued)} (giờ VN) · radar lúc {hhmm(fc.anchor)} · "
                                f"{duration(now - fc.issued)} trước")
        newest = self.store.load(self.issues[-1]) if self.issues else fc
        late = now - newest.issued
        self.stale.setVisible(late > STALE_AFTER)
        self.stale.setText(f"⚠ Không có bản tin mới từ {hhmm(newest.issued)} {day(newest.issued)} ({duration(late)})")
        chips = []
        for source, h in fc.health.items():
            ok = h.complete
            color = GOOD if ok else LEVEL_COLOR[Level.ALERT]
            chips.append(f"<span style='border:1px solid {color}; color:{INK}; padding:2px 6px;'>&nbsp;{'✓' if ok else '⚠'} "
                         f"{SOURCE_NAMES[source]}{'' if ok else f' {h.frames}/{h.needed}'}&nbsp;</span>")
        self.health.setText("&nbsp; ".join(chips))

    def _fill_table(self) -> None:
        fc = self.fc
        if fc is None:
            return
        li, k = fc.level_index, self.k
        prev = {a.code: a for a in self.prev.communes} if self.prev else {}
        province = None if self.province_filter.currentIndex() == 0 else self.provinces[self.province_filter.currentIndex() - 1]
        rows = []
        for a in fc.communes:
            before = prev.get(a.code)
            if province and a.province != province:
                continue
            if self.only_notable.isChecked() and a.code not in self.notable:
                continue
            if self.only_levels.isChecked() and a.level == Level.NONE and (before is None or before.level == Level.NONE):
                continue
            rows.append((a, before))
        rows.sort(key=lambda r: (-LEVEL_RANK[r[0].level], -r[0].probability[li]))
        heads = ["Tỉnh/TP", "Phường/xã", f"Cấp\n{until(fc, li).split(' (')[0]}"] + \
                [window_label(fc, i).replace(" (", "\n").rstrip(")") for i in range(len(fc.windows))] + ["So với\nbản trước"]
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(heads))
        self.table.setHorizontalHeaderLabels(heads)
        self.table.setRowCount(len(rows))
        bold = QFont(self.table.font())
        bold.setBold(True)
        if (head := self.table.horizontalHeaderItem(3 + k)) is not None:
            head.setFont(bold)
        for r, (a, before) in enumerate(rows):
            self.table.setItem(r, 0, QTableWidgetItem(province_short(a.province)))
            name = QTableWidgetItem(a.name + (" ★" if a.code in self.notable else ""))
            name.setData(Qt.ItemDataRole.UserRole, a.code)
            self.table.setItem(r, 1, name)
            lvl = QTableWidgetItem(f"{LEVEL_ICON[a.level]} {LEVEL_TEXT[a.level]}")
            if (c := LEVEL_COLOR[a.level]) is not None:
                lvl.setBackground(QBrush(QColor(c)))
                lvl.setForeground(QBrush(QColor(ink_on(c))))
            self.table.setItem(r, 2, lvl)
            for j, p in enumerate(a.probability):
                cell = QTableWidgetItem(f"{p:.0%}")
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if (c := prob_color(p)) is not None:
                    cell.setBackground(QBrush(QColor(c)))
                    cell.setForeground(QBrush(QColor(ink_on(c))))
                if j == k:
                    cell.setFont(bold)
                self.table.setItem(r, 3 + j, cell)
            self.table.setItem(r, 3 + len(a.probability), QTableWidgetItem(change_text(a, before, li)))
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for col, width in [(0, 104), (2, 86)] + [(3 + j, 56) for j in range(len(fc.windows))] + [(3 + len(fc.windows), 84)]:
            self.table.setColumnWidth(col, width)
        n_warn = sum(a.level == Level.WARNING for a in fc.communes)
        n_alert = sum(a.level == Level.ALERT for a in fc.communes)
        self.table_count.setText(f"{n_warn} cảnh báo · {n_alert} theo dõi · {len(rows)} dòng")

    def _on_row(self) -> None:
        items = self.table.selectedItems()
        code = next((it.data(Qt.ItemDataRole.UserRole) for it in items if it.column() == 1), None)
        self.canvas.highlight_commune(self.geo.by_code.get(code) if code else None)

    def _fill_bulletin(self) -> None:
        if self.fc is not None:
            self.text.setPlainText(bulletin(self.fc, self.geo))

    def _fill_info(self) -> None:
        fc = self.fc
        assert fc is not None
        lv = fc.levels
        windows = "".join(f"<li>({hhmm(fc.anchor + timedelta(minutes=10))}, {hhmm(fc.anchor + timedelta(minutes=w.end_min))}] – "
                          f"bán kính {w.radius_km:g} km</li>" for w in fc.windows)
        health = "".join(f"<li>{SOURCE_NAMES[s]}: {h.frames}/{h.needed} ảnh</li>" for s, h in fc.health.items())
        self.info.setText(
            f"<h3>Sản phẩm</h3><p>Xác suất có ít nhất một cú sét trong bán kính quanh mỗi điểm, trong từng khung thời gian tính "
            f"từ 10 phút sau giờ radar {hhmm(fc.anchor)}. Lưới 2 km quanh radar Nhà Bè.</p><ul>{windows}</ul>"
            f"<h3>Mức</h3><p>{LEVEL_ICON[Level.WARNING]} {LEVEL_TEXT[Level.WARNING]}: ≥ {lv.warning:.0%} · "
            f"{LEVEL_ICON[Level.ALERT]} {LEVEL_TEXT[Level.ALERT]}: ≥ {lv.alert:.0%}, trên khung {lv.window_min - 10} phút, "
            f"bán kính {lv.radius_km:g} km. Mức của một phường/xã là xác suất cao nhất trên các ô lưới của nó: khả năng có sét "
            f"trong bán kính {lv.radius_km:g} km quanh một điểm nào đó của phường/xã. Phường/xã đánh dấu ★ là phường/xã tiêu biểu "
            f"(ghi tên trên bản đồ).</p>"
            f"<h3>Dữ liệu đầu vào</h3><p>{inputs_line(fc)}</p><ul>{health}</ul>"
            f"<h3>Mô hình</h3><p>{fc.model}<br>Phát hành {hhmm(fc.issued)} {day(fc.issued)} (giờ VN).</p>"
            f"<p style='color:{MUTED}'>Thư mục: {self.store.root}</p>")

    # ------------------------------------------------------------------------------------------- export
    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self.text.toPlainText())

    def _save_text(self) -> None:
        if self.fc is None:
            return
        name, _ = QFileDialog.getSaveFileName(self, "Lưu bản tin", f"ban_tin_set_{self.fc.issued + timedelta(hours=7):%Y%m%d_%H%M}.txt", "Văn bản (*.txt)")
        if name:
            Path(name).write_text(self.text.toPlainText(), encoding="utf-8")

    def _save_geojson(self) -> None:
        if self.fc is None:
            return
        name, _ = QFileDialog.getSaveFileName(self, "Xuất GeoJSON", f"canh_bao_set_{self.fc.issued + timedelta(hours=7):%Y%m%d_%H%M}.geojson",
                                              "GeoJSON (*.geojson)")
        if name:
            Path(name).write_text(json.dumps(geojson(self.fc, self.geo), ensure_ascii=False), encoding="utf-8")
