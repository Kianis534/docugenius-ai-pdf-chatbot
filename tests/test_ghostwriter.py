"""Unit tests for the rule-based style-transfer engine."""

from __future__ import annotations

import random

import pytest

from docugenius.errors import LLMNotConfiguredError
from docugenius.services.ghostwriter import (
    STYLES,
    TWITTER_LIMIT,
    Ghostwriter,
    RuleBasedRewriter,
    inflect_verb,
    pluralize,
    truncate_text,
)


@pytest.fixture
def rules() -> RuleBasedRewriter:
    return RuleBasedRewriter(random.Random(1))


@pytest.mark.parametrize(
    ("lemma", "form", "expected"),
    [
        ("utilize", "past", "utilized"),
        ("utilize", "gerund", "utilizing"),
        ("utilize", "third", "utilizes"),
        ("embark upon", "past", "embarked upon"),
        ("verify", "third", "verifies"),
        ("verify", "past", "verified"),
        ("employ", "past", "employed"),
        ("commence", "gerund", "commencing"),
        ("begin", "past", "began"),
        ("begin", "gerund", "beginning"),
    ],
)
def test_inflect_verb(lemma, form, expected):
    assert inflect_verb(lemma, form) == expected


@pytest.mark.parametrize(
    ("noun", "plural"), [("issue", "issues"), ("child", "children"), ("complication", "complications")]
)
def test_pluralize(noun, plural):
    assert pluralize(noun) == plural


def test_need_to_becomes_must(rules):
    assert rules.rewrite("We need to fix this problem. She needed to try again.", "formal") == (
        "We must rectify this issue. She had to attempt again."
    )


def test_formal_preserves_tense_and_case(rules):
    assert rules.rewrite("They used the tool and started early.", "formal") == (
        "They utilized the tool and commenced early."
    )


def test_formal_expands_contractions(rules):
    assert rules.rewrite("We don't think it's ready.", "formal") == "We do not believe it is ready."


def test_casual_contracts_but_not_at_clause_end(rules):
    assert rules.rewrite("I am sure it is fine. That is what it is.", "casual") == (
        "I'm sure it's fine. That's what it is."
    )


def test_nouns_are_pluralised(rules):
    assert rules.rewrite("These problems are big.", "formal") == "These issues are significant."


def test_ambiguous_verbs_need_verb_context(rules):
    # "use" as a noun must not change; as a verb after "to" it should.
    assert rules.rewrite("The use of AI is growing.", "formal") == "The use of AI is growing."
    assert rules.rewrite("We plan to use AI.", "formal") == "We plan to utilize AI."


def test_concise_removes_filler_and_verbose_phrases(rules):
    result = rules.rewrite("In order to win, we really need to act due to the fact that time is short.", "concise")
    assert result == "To win, we need to act since time is short."


def test_twitter_respects_limit(rules):
    long_text = "This is a sentence that keeps going and going. " * 20
    result = rules.rewrite(long_text, "twitter")
    assert len(result) <= TWITTER_LIMIT


def test_creative_is_deterministic_with_seed():
    first = RuleBasedRewriter(random.Random(3)).rewrite("She said it was happy news.", "creative")
    second = RuleBasedRewriter(random.Random(3)).rewrite("She said it was happy news.", "creative")
    assert first == second
    assert "said" not in first


@pytest.mark.parametrize("style", STYLES)
def test_every_style_returns_text(rules, style):
    assert rules.rewrite("We need to check the problem quickly because it is important.", style)


def test_unknown_style_raises(rules):
    with pytest.raises(ValueError):
        rules.rewrite("hello", "pirate")


def test_truncate_prefers_sentence_boundary():
    text = "First sentence is here. Second sentence is a bit longer than the first one."
    assert truncate_text(text, 40) == "First sentence is here."
    assert len(truncate_text("word " * 100, 50)) <= 50


def test_facade_reports_unchanged_text():
    writer = Ghostwriter(llm=None, rules=RuleBasedRewriter(random.Random(0)))
    result = writer.rewrite("Hello there.", "legal")
    assert result.engine == "rules"
    assert result.rewritten == "Hello there."
    assert any("No changes" in w for w in result.warnings)


def test_facade_forced_gemini_without_key_raises():
    with pytest.raises(LLMNotConfiguredError):
        Ghostwriter(llm=None).rewrite("Hello", "formal", engine="gemini")
