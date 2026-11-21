"""Reddit sentiment analysis.

Posts and top comments matching a topic are collected with PRAW (read-only,
application-only OAuth), scored with TextBlob's lexicon-based polarity model,
aggregated, charted with Matplotlib, and finally summarised by Gemini. If
Gemini is unavailable, a deterministic summary is produced instead so the
feature still returns useful output.
"""

from __future__ import annotations

import base64
import logging
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from statistics import fmean
from typing import TYPE_CHECKING, Any, Literal

from docugenius.config import Settings
from docugenius.errors import LLMError, RedditError, RedditNotConfiguredError

if TYPE_CHECKING:
    from docugenius.services.llm import GeminiClient

logger = logging.getLogger(__name__)

TimeFilter = Literal["hour", "day", "week", "month", "year", "all"]
NEUTRAL_BAND = 0.05
COLLECTION_BUDGET_SECONDS = 40.0
SUBMISSIONS_WITH_COMMENTS = 10
COMMENTS_PER_SUBMISSION = 3

SENTIMENT_COLORS = {"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"}

ANALYSIS_SYSTEM_PROMPT = (
    "You are a social media analyst. Write concise, neutral, evidence-based insights. "
    "Do not quote usernames and do not speculate beyond the data you are given."
)


@dataclass
class Post:
    text: str
    user: str
    upvotes: int
    subreddit: str
    url: str
    created_at: str
    is_submission: bool
    polarity: float = 0.0
    sentiment: str = "neutral"


def classify_polarity(polarity: float) -> str:
    if polarity > NEUTRAL_BAND:
        return "positive"
    if polarity < -NEUTRAL_BAND:
        return "negative"
    return "neutral"


class RedditSentimentAnalyzer:
    def __init__(self, settings: Settings, llm: GeminiClient, reddit: Any | None = None) -> None:
        self._settings = settings
        self._llm = llm
        self._reddit = reddit

    @property
    def is_configured(self) -> bool:
        return self._reddit is not None or bool(self._settings.reddit_client_id and self._settings.reddit_client_secret)

    # ------------------------------------------------------------------ #
    def analyze(self, subject: str, time_filter: TimeFilter = "week", limit: int = 25) -> dict[str, Any]:
        posts = self._collect(subject.strip(), time_filter, limit)
        if not posts:
            raise RedditError("No Reddit posts matched this topic in the selected time range.", status_code=404)

        from textblob import TextBlob

        for post in posts:
            post.polarity = round(TextBlob(post.text).sentiment.polarity, 3)
            post.sentiment = classify_polarity(post.polarity)

        counts = Counter(p.sentiment for p in posts)
        total = len(posts)
        percentages = {k: round(100 * counts.get(k, 0) / total, 1) for k in SENTIMENT_COLORS}
        top_subreddits = dict(Counter(p.subreddit for p in posts).most_common(5))
        samples = self._pick_samples(posts)
        average_polarity = round(fmean(p.polarity for p in posts), 3)

        stats = {
            "subject": subject,
            "time_filter": time_filter,
            "total_posts": total,
            "submissions": sum(p.is_submission for p in posts),
            "comments": sum(not p.is_submission for p in posts),
            **percentages,
            "average_polarity": average_polarity,
            "average_upvotes": round(fmean(p.upvotes for p in posts), 1),
            "overall": overall_sentiment(percentages),
            "top_subreddits": top_subreddits,
        }
        analysis, engine = self._summarise(stats, samples)
        return {
            **stats,
            "chart": render_sentiment_chart(counts, subject),
            "sample_posts": [self._public(p) for p in samples],
            "analysis": analysis,
            "analysis_engine": engine,
        }

    # ------------------------------------------------------------------ #
    def _client(self) -> Any:
        if self._reddit is None:
            if not self.is_configured:
                raise RedditNotConfiguredError()
            import praw

            self._reddit = praw.Reddit(
                client_id=self._settings.reddit_client_id,
                client_secret=self._settings.reddit_client_secret,
                user_agent=self._settings.reddit_user_agent,
                check_for_async=False,
                timeout=20,
            )
            self._reddit.read_only = True
        return self._reddit

    def _collect(self, subject: str, time_filter: str, limit: int) -> list[Post]:
        import prawcore

        reddit = self._client()
        started = time.monotonic()
        posts: list[Post] = []
        try:
            submissions = reddit.subreddit("all").search(
                subject, sort="relevance", time_filter=time_filter, limit=limit
            )
            for index, submission in enumerate(submissions):
                if time.monotonic() - started > COLLECTION_BUDGET_SECONDS:
                    logger.info("Reddit collection budget reached after %d submissions", index)
                    break
                posts.append(self._to_post(submission, is_submission=True))
                if index < SUBMISSIONS_WITH_COMMENTS:
                    posts.extend(self._top_comments(submission))
        except prawcore.exceptions.OAuthException as exc:
            raise RedditError(
                "Reddit rejected the credentials. Check REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."
            ) from exc
        except prawcore.exceptions.ResponseException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 401:
                raise RedditError(
                    "Reddit rejected the credentials. Check REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."
                ) from exc
            if status == 429:
                raise RedditError("Reddit is rate limiting requests. Please wait a minute.", status_code=429) from exc
            raise RedditError() from exc
        except prawcore.exceptions.PrawcoreException as exc:
            logger.warning("Reddit request failed: %s", exc)
            if not posts:
                raise RedditError() from exc
        return posts

    def _top_comments(self, submission: Any) -> list[Post]:
        try:
            submission.comment_sort = "top"
            submission.comments.replace_more(limit=0)
            return [
                self._to_post(comment, is_submission=False, subreddit=submission.subreddit.display_name)
                for comment in list(submission.comments)[:COMMENTS_PER_SUBMISSION]
                if getattr(comment, "body", "") not in ("", "[deleted]", "[removed]")
            ]
        except Exception as exc:  # comments are best-effort
            logger.debug("Skipping comments for %s: %s", getattr(submission, "id", "?"), exc)
            return []

    @staticmethod
    def _to_post(item: Any, *, is_submission: bool, subreddit: str | None = None) -> Post:
        text = f"{item.title}\n\n{item.selftext or ''}".strip() if is_submission else item.body
        return Post(
            text=text,
            user=str(item.author) if item.author else "[deleted]",
            upvotes=int(item.score or 0),
            subreddit=subreddit or item.subreddit.display_name,
            url=f"https://www.reddit.com{item.permalink}",
            created_at=datetime.fromtimestamp(item.created_utc, tz=UTC).isoformat(timespec="seconds"),
            is_submission=is_submission,
        )

    @staticmethod
    def _pick_samples(posts: list[Post]) -> list[Post]:
        submissions = sorted((p for p in posts if p.is_submission), key=lambda p: p.upvotes, reverse=True)
        comments = sorted((p for p in posts if not p.is_submission), key=lambda p: p.upvotes, reverse=True)
        return submissions[:3] + comments[:2]

    @staticmethod
    def _public(post: Post) -> dict[str, Any]:
        text = post.text if len(post.text) <= 600 else post.text[:597].rsplit(" ", 1)[0] + "…"
        return {
            "text": text,
            "user": post.user,
            "upvotes": post.upvotes,
            "subreddit": post.subreddit,
            "url": post.url,
            "created_at": post.created_at,
            "is_submission": post.is_submission,
            "sentiment": post.sentiment,
            "polarity": post.polarity,
        }

    def _summarise(self, stats: dict[str, Any], samples: list[Post]) -> tuple[str, str]:
        if self._llm.is_configured:
            excerpts = "\n\n".join(
                f"- r/{p.subreddit} ({p.upvotes} upvotes, {p.sentiment}): {p.text[:500]}" for p in samples
            )
            prompt = (
                f"Topic: {stats['subject']} (Reddit, time range: {stats['time_filter']})\n"
                f"Posts analysed: {stats['total_posts']}\n"
                f"Sentiment: {stats['positive']}% positive, {stats['neutral']}% neutral, "
                f"{stats['negative']}% negative (mean polarity {stats['average_polarity']})\n"
                f"Most active subreddits: {', '.join(stats['top_subreddits'])}\n\n"
                f"Representative posts:\n{excerpts}\n\n"
                "Write 3-5 bullet points describing the overall mood, the main themes driving "
                "positive and negative opinions, and anything notable. Finish with a one-line verdict."
            )
            try:
                return self._llm.generate(prompt, system_instruction=ANALYSIS_SYSTEM_PROMPT).text, "gemini"
            except LLMError as exc:
                logger.warning("Falling back to heuristic Reddit summary: %s", exc.message)
        return heuristic_summary(stats), "heuristic"


def overall_sentiment(percentages: dict[str, float]) -> str:
    positive, negative = percentages.get("positive", 0), percentages.get("negative", 0)
    if positive >= 60:
        return "Strongly positive"
    if negative >= 60:
        return "Strongly negative"
    if positive - negative >= 15:
        return "Mostly positive"
    if negative - positive >= 15:
        return "Mostly negative"
    return "Mixed"


def heuristic_summary(stats: dict[str, Any]) -> str:
    subreddits = ", ".join(f"r/{name}" for name in stats["top_subreddits"]) or "n/a"
    return (
        f"- **Overall mood:** {stats['overall']} across {stats['total_posts']} posts and comments.\n"
        f"- **Split:** {stats['positive']}% positive, {stats['neutral']}% neutral, {stats['negative']}% negative.\n"
        f"- **Mean polarity:** {stats['average_polarity']} on a scale from -1 (negative) to +1 (positive).\n"
        f"- **Where it is discussed:** {subreddits}.\n"
        f"- **Engagement:** {stats['average_upvotes']} upvotes on average."
    )


def render_sentiment_chart(counts: Counter[str], subject: str) -> str | None:
    """Render a donut chart and return it as a ``data:image/png;base64`` URI."""
    try:
        from matplotlib.figure import Figure  # object-oriented API: thread-safe, no pyplot state
    except ImportError:  # pragma: no cover
        return None

    labels = [k for k in SENTIMENT_COLORS if counts.get(k)]
    if not labels:
        return None
    figure = Figure(figsize=(5, 5), dpi=110)
    axes = figure.subplots()
    axes.pie(
        [counts[k] for k in labels],
        labels=[k.capitalize() for k in labels],
        colors=[SENTIMENT_COLORS[k] for k in labels],
        autopct="%1.0f%%",
        startangle=90,
        pctdistance=0.78,
        wedgeprops={"width": 0.45, "edgecolor": "white"},
        textprops={"fontsize": 11},
    )
    axes.set_title(f"Reddit sentiment: {subject}"[:60], fontsize=12)
    axes.axis("equal")
    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", transparent=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
