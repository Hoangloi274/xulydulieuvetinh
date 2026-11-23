"""Picks the live forecasts worth showing (strong signal or many strokes) and renders the operator app for them,
once the verifying strokes are in (the ✕ marks: strokes that then fell in the window).

    uv run --with selenium python scripts/capture_moments.py --work-dir live [--min-prob 0.3] [--min-strokes 30]

Writes <work-dir>/moments/moments.csv (every forecast scored) and, per flagged forecast,
<work-dir>/moments/YYYYmmdd/HHMM_{radar,p60,p120}.png (anchor T in UTC) and HHMM_panel.png (scripts/moment_panel.py:
made at T+70, remade once hymetnet's frame of the strokes is saved). Prints one line per flagged forecast.

Also archives hymetnet's lightning page (strokes embedded in the HTML) and its thunderstorm warnings (/dongset)
every 5 min to <work-dir>/hymetnet/YYYYmmdd/HHMM{.html,_dongset.json} (fetch time, UTC), and renders it with
headless Firefox, captioned with the times its data cover: HHMM_hm.png, the page at T+5 (the warnings it had issued;
its strokes lag ~1 h), and HHMM_hm_verify.png, the first saved page whose frame holds the strokes up to T+60.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

# when each picture can be made: the window's strokes reach the app with the forecast issued after its end
SHOTS = {"radar": (0, ["--radar"]), "p60": (70, ["--window", "60"]), "p120": (130, ["--window", "120"])}
HYMETNET = "http://hymetnet.gov.vn"
HM_EVERY = timedelta(minutes=5)
VN = timedelta(hours=7)
# hymetnet's observed frame labelled L (set[6..17]) maps the strokes of (L-50, L+10]; its stroke frames lag 51-63 min
# behind the page, its warning cells (set[0..5], /dongset) do not. The strokes of the 60-min window (T+10, T+60] are
# in frame L = T+50, on a page fetched near T+115-125.
HM_VERIFY_FRAME = timedelta(minutes=50)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def score(folder: Path) -> tuple[datetime, float, int]:
    """Anchor, highest commune probability in the levels' window (regions, in forecasts written before the communes),
    strokes in the 30 min up to T+5."""
    d = json.loads((folder / "forecast.json").read_text(encoding="utf-8"))
    k = next(i for i, w in enumerate(d["windows"]) if w["end_min"] == d["levels"]["window_min"])
    prob = max((r["probability"][k] for r in d.get("communes", d.get("regions", ()))), default=0.0)
    with np.load(folder / "observed.npz") as z:
        strokes = int((z["stroke_min"] > -25).sum())
    return datetime.fromisoformat(d["anchor"]), prob, strokes


def shoot(work_dir: Path, anchor: datetime, out: Path, extra: list[str]) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "lightning_nowcast.app", "--work-dir", str(work_dir), "--issue", anchor.isoformat(),
           "--screenshot", str(out), *extra]
    try:  # generous: a busy drive (bulk downloads on the same disk) slows the app's reads a lot
        r = subprocess.run(cmd, env={**os.environ, "QT_QPA_PLATFORM": "offscreen"}, capture_output=True, text=True, timeout=600, check=False)
    except subprocess.TimeoutExpired:
        print(f"app timed out on {out}", file=sys.stderr)
        return False
    return r.returncode == 0 and out.exists()


def fetch_hymetnet(root: Path, now: datetime) -> None:
    """The page and the warnings for the current 5-min slot, unless already saved."""
    t = datetime.min + (now - datetime.min) // HM_EVERY * HM_EVERY
    for url, out in ((f"{HYMETNET}/lightningmaps/", root / f"{t:%Y%m%d}" / f"{t:%H%M}.html"),
                     (f"{HYMETNET}/dongset", root / f"{t:%Y%m%d}" / f"{t:%H%M}_dongset.json")):
        if out.exists():
            continue
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                body = r.read()
        except Exception as exc:  # noqa: BLE001 - outage: the next poll tries again
            print(f"hymetnet {url}: {exc}", file=sys.stderr)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".tmp").write_bytes(body)
        out.with_suffix(".tmp").replace(out)


def hymetnet_page(root: Path, t: datetime, within: timedelta = timedelta(minutes=15)) -> Path | None:
    """The first saved page fetched at or after t (within `within`)."""
    first = datetime.min + -(-(t - datetime.min) // HM_EVERY) * HM_EVERY  # t rounded up to a fetch slot
    for k in range(int(within / HM_EVERY) + 1):
        f = first + k * HM_EVERY
        p = root / f"{f:%Y%m%d}" / f"{f:%H%M}.html"
        if p.exists():
            return p
    return None


def fetch_time(page: Path) -> datetime:
    return datetime.strptime(page.parent.name + page.stem, "%Y%m%d%H%M")


def page_frames(page: Path) -> tuple[dict[datetime, tuple[int, str]], datetime | None]:
    """The page's observed frames {label (UTC): (set index, label text)} and the time its warnings were issued (UTC)."""
    frames, issued = {}, None
    text = page.read_text(encoding="utf-8", errors="replace")
    for i, label in re.findall(r'tentimesett\[(\d+)\] = "([^"]+)";', text):
        m = re.match(r"(\d\d:\d\d)( \+\d+min)? \(VN\) (\d\d-\d\d-\d{4})", label)
        if m is None:
            continue
        t = datetime.strptime(f"{m[3]} {m[1]}", "%d-%m-%Y %H:%M") - VN
        if m[2]:
            issued = t
        else:
            frames[t] = (int(i), label)
    return frames, issued


def verify_page(root: Path, anchor: datetime) -> tuple[Path, int, str] | None:
    """The first saved page holding the frame of the strokes up to T+60."""
    target = anchor + HM_VERIFY_FRAME
    for page in sorted(root.glob("*/[0-9][0-9][0-9][0-9].html")):
        if fetch_time(page) >= anchor + timedelta(minutes=60) and target in (frames := page_frames(page)[0]):
            return page, *frames[target]
    return None


def caption(png: Path, lines: list[str]) -> None:
    """A band above the picture with what it shows and when (times in giờ VN)."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(png).convert("RGB")
    font = ImageFont.truetype(FONT, 22)
    band = 16 + 30 * len(lines)
    out = Image.new("RGB", (im.width, im.height + band), "#1d2733")
    out.paste(im, (0, band))
    draw = ImageDraw.Draw(out)
    for k, line in enumerate(lines):
        draw.text((16, 10 + 30 * k), line, font=font, fill="white" if k == 0 else "#ffd166")
    out.save(png)


def hm_caption(page: Path, frame: tuple[int, str] | None) -> list[str]:
    frames, issued = page_frames(page)
    label_t = max(frames) if frame is None else next(t for t, f in frames.items() if f[0] == frame[0])
    fetched = fetch_time(page)
    lag = int((fetched - (label_t + timedelta(minutes=10))).total_seconds() // 60)

    def vn(t: datetime) -> str:
        return (t + VN).strftime("%H:%M")

    lines = [f"hymetnet.gov.vn/lightningmaps · trang tải lúc {vn(fetched)} {(fetched + VN):%d/%m/%Y} (giờ VN)",
             f"Sét trên bản đồ: {vn(label_t - timedelta(minutes=50))}–{vn(label_t + timedelta(minutes=10))}"
             + (f" (khung mới nhất, chậm {lag} phút so với lúc tải trang)" if frame is None else " (khung đã chọn)")]
    if issued is not None:
        lines.append(f"Cảnh báo dông sét: phát lúc {vn(issued)}, hạn +10…+60 phút")
    return lines


def shoot_hymetnet(page: Path, out: Path, frame: tuple[int, str] | None = None) -> bool:
    """hymetnet's page as it was fetched, its map zoomed on Nam Bộ, via selenium (firefox's own --screenshot shoots
    before the marker icons are drawn). The copy drops the facebook / counter / socket scripts, which keep the page
    loading for ever, and its warning panel reads the /dongset saved with the page instead of the live one.
    `frame`: (set index, label) of the observed frame to show instead of the newest. The picture gets a caption."""
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options

    s = page.read_text(encoding="utf-8", errors="replace")
    s = re.sub(r"<script[^>]*(facebook|socket\.io|histats)[^>]*>\s*</script>", "", s)
    s = re.sub(r'<meta http-equiv="refresh"[^>]*>', "", s)
    s = s.replace("<head>", f'<head><base href="{HYMETNET}/lightningmaps/">', 1)
    dongset = page.with_name(page.stem + "_dongset.json")
    warnings = dongset.read_text(encoding="utf-8") if dongset.exists() else "{}"
    s = re.sub(r'(<script[^>]*axios[^>]*></script>)', lambda m: m.group(1) + (
        "<script>(function () { var saved = " + warnings + ", create = axios.create;"
        " axios.create = function (c) { var api = create(c);"
        " api.get = function (url) { return url.indexOf('/dongset') >= 0 ? Promise.resolve({data: saved}) : Promise.reject(url); };"
        " return api; }; })();</script>"), s, count=1)
    # a frame is chosen as a user does, through the time selector: its list runs the other way to set[]
    show = "" if frame is None else (f"window.addEventListener('load', function () {{ setTimeout(function () {{"
                                     f" var e = document.getElementById('time'); e.value = {json.dumps(frame[1])};"
                                     f" e.dispatchEvent(new Event('change')); }}, 300); }});")
    s = s.replace("</body>", f"<script>map.setView([10.55, 106.45], 8); {show}</script></body>", 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    opts = Options()
    for pref in ("security.mixed_content.upgrade_display_content", "dom.security.https_first", "dom.security.https_first_pbm"):
        opts.set_preference(pref, False)  # hymetnet is http only: firefox would upgrade its icons to https and lose them
    for arg in ("--headless", "--width=1600", "--height=1000"):
        opts.add_argument(arg)
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "page.html"
        html.write_text(s, encoding="utf-8")
        try:
            driver = webdriver.Firefox(options=opts)
        except Exception as exc:  # noqa: BLE001 - no firefox / geckodriver: no hymetnet pictures
            print(f"firefox: {exc}", file=sys.stderr)
            return False
        try:
            driver.set_page_load_timeout(60)
            driver.get(html.as_uri())
            wait_images = ("return Array.from(document.images).filter(i => i.src.includes('hymetnet') || i.classList.contains('leaflet-tile'))"
                           ".every(i => i.complete && i.naturalWidth > 0)")
            deadline = time.time() + 30
            while not driver.execute_script(wait_images) and time.time() < deadline:
                time.sleep(0.5)
            time.sleep(1.5)  # the warning panel, the chart and the chosen frame
            driver.save_screenshot(str(out.resolve()))
            caption(out, hm_caption(page, frame))
        except Exception as exc:  # noqa: BLE001 - a slow or broken page: retried on the next poll
            print(f"firefox failed on {page}: {exc}", file=sys.stderr)
        finally:
            driver.quit()
    return out.exists()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--min-prob", type=float, default=0.3, help="Flag when a commune reaches this probability")
    ap.add_argument("--min-strokes", type=int, default=30, help="... or the last 30 min had this many strokes")
    ap.add_argument("--poll-sec", type=float, default=60.0)
    args = ap.parse_args()
    root = args.work_dir / "moments"
    root.mkdir(parents=True, exist_ok=True)
    log = root / "moments.csv"
    seen: dict[datetime, bool] = {}
    failed: set[Path] = set()  # given up on
    tries: dict[Path, int] = {}
    if log.exists():
        with log.open() as f:
            seen = {datetime.fromisoformat(r["anchor"]): r["flagged"] == "1" for r in csv.DictReader(f)}
    else:
        log.write_text("anchor,max_prob,strokes30,flagged\n")
    hm_root = args.work_dir / "hymetnet"
    while True:
        fetch_hymetnet(hm_root, utcnow())
        for fj in sorted(args.work_dir.glob("forecasts/*/*/forecast.json")):
            try:
                anchor, prob, strokes = score(fj.parent)
            except Exception as exc:  # noqa: BLE001 - a folder being written: next poll
                print(f"skip {fj.parent}: {exc}", file=sys.stderr)
                continue
            if anchor in seen:
                continue
            flagged = prob >= args.min_prob or strokes >= args.min_strokes
            seen[anchor] = flagged
            with log.open("a") as f:
                f.write(f"{anchor.isoformat()},{prob:.3f},{strokes},{int(flagged)}\n")
            if flagged:
                local = anchor + timedelta(hours=7)
                print(f"MOMENT {anchor:%Y-%m-%d %H:%M} UTC ({local:%H:%M} giờ VN): max region {prob:.0%}, {strokes} strokes in 30 min", flush=True)
        now = utcnow()
        for anchor, flagged in sorted(seen.items()):
            if not flagged:
                continue
            for name, (after_min, extra) in SHOTS.items():
                out = root / f"{anchor:%Y%m%d}" / f"{anchor:%H%M}_{name}.png"
                if not out.exists() and out not in failed and now >= anchor + timedelta(minutes=after_min):
                    ok = shoot(args.work_dir, anchor, out, extra)
                    if not ok:
                        failed.add(out)
                    print(f"{'saved' if ok else 'FAILED'} {out}", flush=True)
            # hymetnet at issue (its newest frame and the warnings it had issued), then the frame of the strokes up to T+60
            for name in ("hm", "hm_verify"):
                out = root / f"{anchor:%Y%m%d}" / f"{anchor:%H%M}_{name}.png"
                if out.exists() or out in failed:
                    continue
                if name == "hm":
                    if now < anchor + timedelta(minutes=5):
                        continue
                    page, frame = hymetnet_page(hm_root, anchor + timedelta(minutes=5)), None
                    late = now >= anchor + timedelta(minutes=25)
                else:
                    found = verify_page(hm_root, anchor)
                    page, frame = (found[0], found[1:]) if found else (None, None)
                    late = now >= anchor + timedelta(minutes=180)
                if page is None and not late:
                    continue  # not fetched yet
                ok = page is not None and shoot_hymetnet(page, out, frame)
                tries[out] = tries.get(out, 0) + 1
                if not ok and page is not None and tries[out] < 3:
                    continue  # firefox fails now and then: next poll
                if not ok:
                    failed.add(out)
                print(f"{'saved' if ok else 'FAILED'} {out}" + ("" if page else " (no hymetnet page with that data)"), flush=True)

            panel = root / f"{anchor:%Y%m%d}" / f"{anchor:%H%M}_panel.png"
            verify = root / f"{anchor:%Y%m%d}" / f"{anchor:%H%M}_hm_verify.png"
            stale = panel.exists() and verify.exists() and panel.stat().st_mtime < verify.stat().st_mtime
            if (stale or not panel.exists()) and panel not in failed and now >= anchor + timedelta(minutes=70):
                r = subprocess.run([sys.executable, str(Path(__file__).with_name("moment_panel.py")), "--work-dir", str(args.work_dir),
                                    anchor.isoformat()], capture_output=True, text=True, timeout=300, check=False)
                if r.returncode or not panel.exists():
                    failed.add(panel)
                    print(f"FAILED {panel}: {r.stderr.strip()[-300:]}", flush=True)
                else:
                    print(r.stdout.strip(), flush=True)
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
