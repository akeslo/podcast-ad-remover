"""Tests for app/core/rss_gen.py: CDATA safety and generated feed validity."""
import os
import xml.etree.ElementTree as ET
from types import SimpleNamespace

from app.core import rss_gen
from app.core.rss_gen import RSSGenerator, _cdata


def test_cdata_wraps_plain_text():
    assert _cdata("hello") == "<![CDATA[hello]]>"


def test_cdata_none_is_empty_section():
    assert _cdata(None) == "<![CDATA[]]>"


def test_cdata_neutralises_terminator_and_opener():
    out = _cdata("a ]]> b <![CDATA[ c")
    inner = out[len("<![CDATA["):-len("]]>")]
    assert "]]>" not in inner
    assert "<![CDATA[" not in inner


def _generator(tmp_path, monkeypatch, descriptions, sub_description="About"):
    podcasts = tmp_path / "podcasts"
    feeds = tmp_path / "feeds"
    feeds.mkdir()
    monkeypatch.setattr(
        rss_gen, "settings",
        SimpleNamespace(PODCASTS_DIR=str(podcasts), FEEDS_DIR=str(feeds), BASE_URL="http://host:8000"),
    )

    sub = SimpleNamespace(
        title="Show", description=sub_description, feed_url="http://x/feed",
        image_url="http://x/img.png", slug="show",
    )
    episodes = []
    for i, d in enumerate(descriptions):
        episodes.append({
            "title": f"Ep {i}", "guid": f"g{i}", "pub_date": "2026-01-01T00:00:00",
            "duration": 60, "local_filename": str(podcasts / "show" / f"e{i}" / "a.mp3"),
            "file_size": 100, "is_video": False, "ai_summary": None,
            "description": d, "original_url": "http://orig",
        })

    gen = RSSGenerator.__new__(RSSGenerator)
    gen.sub_repo = SimpleNamespace(get_by_id=lambda _id: sub)
    gen.ep_repo = SimpleNamespace(get_processed_by_subscription=lambda _id: episodes)
    monkeypatch.setattr(gen, "_get_base_url", lambda: "http://host:8000")
    return gen


def test_generate_feed_missing_subscription_returns_none(tmp_path, monkeypatch):
    gen = _generator(tmp_path, monkeypatch, [])
    gen.sub_repo = SimpleNamespace(get_by_id=lambda _id: None)
    assert gen.generate_feed(1) is None


def test_generate_feed_is_well_formed_with_hostile_descriptions(tmp_path, monkeypatch):
    gen = _generator(tmp_path, monkeypatch, ["bad ]]> end", "open <![CDATA[ x", "a &amp; b <b>x</b>"])
    path = gen.generate_feed(1)
    tree = ET.parse(path)  # raises on malformed XML
    items = tree.getroot().findall("./channel/item")
    assert len(items) == 3
    assert all(i.find("description").text for i in items)


def test_generate_feed_enclosure_url_uses_relative_path(tmp_path, monkeypatch):
    gen = _generator(tmp_path, monkeypatch, ["d"])
    path = gen.generate_feed(1)
    enc = ET.parse(path).getroot().find("./channel/item/enclosure")
    assert enc.get("url") == "http://host:8000/audio/show/e0/a.mp3"
    assert enc.get("type") == "audio/mpeg"
    assert enc.get("length") == "100"
    assert os.path.basename(path) == "show.xml"


def test_generate_feed_missing_description_falls_back_to_original_url(tmp_path, monkeypatch):
    gen = _generator(tmp_path, monkeypatch, [None])
    path = gen.generate_feed(1)
    desc = ET.parse(path).getroot().find("./channel/item/description").text
    assert "http://orig" in desc
