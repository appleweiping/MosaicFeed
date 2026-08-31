"""Portable, dependency-free HTML rendering for a generated feed."""

from __future__ import annotations

import html
from collections.abc import Mapping
from pathlib import Path

from mosaicfeed.models import Article, Feed


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def render_feed_report(
    feed: Feed,
    articles: Mapping[str, Article],
    *,
    output: str | Path,
) -> None:
    """Render a self-contained, script-free review dashboard."""

    source_count = len({articles[item.article_id].source for item in feed.recommendations})
    topic_count = len(
        {
            topic
            for item in feed.recommendations
            for topic in articles[item.article_id].topics
        }
    )
    cards: list[str] = []
    for item in feed.recommendations:
        article = articles[item.article_id]
        breakdown = item.breakdown
        components = (
            ("interest", breakdown.interest),
            ("freshness", breakdown.freshness),
            ("quality", breakdown.quality),
            ("novelty", breakdown.novelty),
            ("popularity", breakdown.popularity),
        )
        bars = "".join(
            f'<div class="metric"><span>{_escape(name)}</span><div class="track"><i style="width:{value * 100:.1f}%"></i></div><b>{value:.2f}</b></div>'
            for name, value in components
        )
        reasons = "".join(f"<li>{_escape(reason)}</li>" for reason in breakdown.reasons)
        topics = "".join(
            f'<span class="chip">{_escape(topic)}</span>' for topic in article.topics
        )
        cards.append(
            f"""
            <article class="card">
              <div class="rank">#{item.rank}</div>
              <div class="body">
                <div class="eyebrow">{_escape(article.source)} · {_escape(article.published_at.date())}</div>
                <h2>{_escape(article.title)}</h2>
                <p>{_escape(article.summary)}</p>
                <div class="chips">{topics}</div>
                <div class="scoreline"><strong>{item.score:.3f}</strong><span>ranking score before diversity reranking</span></div>
                <div class="metrics">{bars}</div>
                <ul>{reasons}</ul>
              </div>
            </article>
            """
        )
    body = "".join(cards) or '<div class="empty">No eligible candidates met the configured constraints.</div>'
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>MosaicFeed report for {_escape(feed.user_id)}</title>
  <style>
    :root{{--ink:#10243e;--muted:#5b6980;--line:#d9e3ec;--paper:#f3f7f8;--card:#fff;--teal:#087f8c;--mint:#46c2a6;--amber:#f2b84b}}
    *{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(145deg,#eef6f5,#f7f4ee);color:var(--ink);font:15px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
    main{{max-width:1080px;margin:0 auto;padding:52px 28px 80px}} .top{{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:28px}}
    .kicker{{color:var(--teal);font-size:13px;font-weight:800;letter-spacing:.16em;text-transform:uppercase}} h1{{font-family:Georgia,serif;font-size:42px;line-height:1.04;margin:9px 0 4px}} .sub{{color:var(--muted)}}
    .summary{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}} .stat{{background:rgba(255,255,255,.8);border:1px solid var(--line);border-radius:16px;padding:18px}} .stat b{{display:block;font:700 27px Georgia,serif}} .stat span{{color:var(--muted);font-size:13px}}
    .card{{display:grid;grid-template-columns:64px 1fr;background:var(--card);border:1px solid var(--line);border-radius:20px;padding:24px;margin:15px 0;box-shadow:0 12px 30px rgba(16,36,62,.055)}}
    .rank{{height:42px;width:42px;border-radius:13px;background:#dff5f1;color:var(--teal);display:grid;place-items:center;font-weight:800}} .eyebrow{{color:var(--teal);font-size:12px;font-weight:750;text-transform:uppercase;letter-spacing:.08em}} h2{{font:700 23px/1.18 Georgia,serif;margin:5px 0}} p{{color:var(--muted);margin:0 0 11px}}
    .chips{{display:flex;gap:6px;flex-wrap:wrap}} .chip{{border:1px solid #b8ddd8;background:#f0fbf8;padding:3px 9px;border-radius:999px;font-size:12px}} .scoreline{{display:flex;gap:10px;align-items:baseline;margin:15px 0 8px}} .scoreline strong{{font:700 25px Georgia,serif}} .scoreline span{{color:var(--muted);font-size:12px}}
    .metrics{{display:grid;grid-template-columns:1fr 1fr;gap:8px 25px}} .metric{{display:grid;grid-template-columns:72px 1fr 33px;gap:8px;align-items:center;font-size:11px;color:var(--muted)}} .metric b{{color:var(--ink)}} .track{{height:6px;background:#e8eef2;border-radius:9px;overflow:hidden}} .track i{{display:block;height:100%;background:linear-gradient(90deg,var(--teal),var(--mint))}} ul{{color:var(--muted);padding-left:18px;margin-bottom:0}} .empty{{padding:40px;background:white;border:1px solid var(--line);border-radius:20px}}
    @media(max-width:700px){{.top{{display:block}}.summary{{grid-template-columns:1fr}}.card{{grid-template-columns:1fr}}.rank{{margin-bottom:12px}}.metrics{{grid-template-columns:1fr}}h1{{font-size:34px}}}}
  </style>
</head>
<body><main>
  <div class="top"><div><div class="kicker">Explainable · diverse · offline</div><h1>MosaicFeed</h1><div class="sub">Feed for <strong>{_escape(feed.user_id)}</strong> · {_escape(feed.generated_at.isoformat())}</div></div></div>
  <section class="summary"><div class="stat"><b>{len(feed.recommendations)}</b><span>ranked items</span></div><div class="stat"><b>{source_count}</b><span>distinct sources</span></div><div class="stat"><b>{topic_count}</b><span>topics represented</span></div></section>
  {body}
</main></body></html>"""
    document = "\n".join(line.rstrip() for line in document.splitlines()) + "\n"
    Path(output).write_text(document, encoding="utf-8", newline="\n")
