"""Ghostwriter: rewrite text in a different register (style transfer).

Two engines are available:

* ``gemini`` asks the LLM to rewrite the text in the requested style.
* ``rules`` is a deterministic, dependency-free lexical rewriter built on
  part-of-speech aware synonym substitution with verb inflection,
  contraction handling and phrase-level transforms. It is used whenever
  Gemini is not configured or temporarily unavailable, so the feature keeps
  working offline.
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from docugenius.errors import LLMError, LLMNotConfiguredError

if TYPE_CHECKING:
    from docugenius.services.llm import GeminiClient

logger = logging.getLogger(__name__)

Style = Literal["formal", "concise", "casual", "academic", "legal", "technical", "creative", "twitter"]
Engine = Literal["auto", "gemini", "rules"]

STYLES: tuple[str, ...] = (
    "formal",
    "concise",
    "casual",
    "academic",
    "legal",
    "technical",
    "creative",
    "twitter",
)
TWITTER_LIMIT = 280

STYLE_GUIDES: dict[str, str] = {
    "formal": "a formal, professional register suitable for business correspondence",
    "concise": "a concise version that keeps every key point but uses as few words as possible",
    "casual": "a relaxed, conversational tone, like a message to a friend",
    "academic": "an objective academic register with precise, scholarly wording",
    "legal": "the precise and unambiguous register of a legal document",
    "technical": "clear technical documentation language with precise terminology",
    "creative": "vivid, expressive and engaging prose",
    "twitter": f"a punchy social media post of at most {TWITTER_LIMIT} characters",
}

# --------------------------------------------------------------------------- #
# Lexicon
# --------------------------------------------------------------------------- #
# lemma -> (part of speech, ambiguous with a noun?, {style: replacement})
_LEXICON: dict[str, tuple[str, bool, dict[str, str]]] = {
    # Verbs
    "use": (
        "verb",
        True,
        {"formal": "utilize", "legal": "utilize", "technical": "employ", "academic": "employ", "creative": "wield"},
    ),
    "start": (
        "verb",
        True,
        {
            "formal": "commence",
            "legal": "commence",
            "academic": "initiate",
            "creative": "embark upon",
            "concise": "begin",
        },
    ),
    "begin": ("verb", False, {"formal": "commence", "legal": "commence", "academic": "initiate"}),
    "think": ("verb", False, {"formal": "believe", "legal": "believe", "academic": "contend", "casual": "reckon"}),
    "help": ("verb", True, {"formal": "assist", "legal": "assist", "academic": "assist", "technical": "support"}),
    "buy": ("verb", False, {"formal": "purchase", "legal": "purchase", "academic": "acquire", "casual": "grab"}),
    "need": ("verb", True, {"formal": "require", "legal": "require", "technical": "require", "academic": "require"}),
    "try": ("verb", True, {"formal": "attempt", "legal": "endeavor", "technical": "attempt", "academic": "attempt"}),
    "tell": ("verb", False, {"formal": "inform", "legal": "notify", "academic": "inform"}),
    "fix": ("verb", True, {"formal": "rectify", "technical": "resolve", "academic": "rectify"}),
    "check": ("verb", True, {"formal": "review", "technical": "verify", "legal": "verify", "academic": "examine"}),
    "change": (
        "verb",
        True,
        {"formal": "alter", "legal": "amend", "technical": "modify", "academic": "modify", "creative": "transform"},
    ),
    "end": ("verb", True, {"formal": "conclude", "legal": "terminate", "academic": "conclude"}),
    # Nouns
    "problem": (
        "noun",
        False,
        {
            "formal": "issue",
            "legal": "matter",
            "technical": "defect",
            "academic": "complication",
            "creative": "challenge",
            "casual": "snag",
        },
    ),
    "solution": ("noun", False, {"legal": "resolution", "technical": "fix", "formal": "remedy", "creative": "answer"}),
    "job": ("noun", False, {"formal": "position", "legal": "engagement", "academic": "occupation"}),
    "kid": ("noun", False, {"formal": "child", "legal": "minor", "academic": "child"}),
    # Adjectives
    "good": (
        "adj",
        False,
        {"formal": "satisfactory", "legal": "sufficient", "technical": "effective", "academic": "adequate"},
    ),
    "bad": (
        "adj",
        False,
        {
            "formal": "unsatisfactory",
            "legal": "deficient",
            "technical": "defective",
            "academic": "inadequate",
            "casual": "lame",
        },
    ),
    "big": (
        "adj",
        False,
        {
            "formal": "significant",
            "legal": "substantial",
            "technical": "large",
            "academic": "substantial",
            "creative": "enormous",
            "casual": "huge",
        },
    ),
    "small": (
        "adj",
        False,
        {"formal": "modest", "legal": "minimal", "technical": "minor", "academic": "limited", "creative": "tiny"},
    ),
    "important": (
        "adj",
        False,
        {
            "formal": "significant",
            "legal": "material",
            "technical": "critical",
            "academic": "salient",
            "creative": "vital",
            "concise": "key",
        },
    ),
    # Adverbs and connectives
    "quickly": (
        "adv",
        False,
        {
            "formal": "promptly",
            "legal": "expeditiously",
            "technical": "rapidly",
            "academic": "rapidly",
            "creative": "swiftly",
            "concise": "fast",
        },
    ),
    "slowly": (
        "adv",
        False,
        {"formal": "gradually", "technical": "progressively", "academic": "gradually", "creative": "leisurely"},
    ),
    "soon": ("adv", False, {"formal": "shortly", "legal": "forthwith", "academic": "in due course"}),
    "because": (
        "conn",
        False,
        {
            "formal": "since",
            "legal": "inasmuch as",
            "academic": "owing to the fact that",
            "concise": "since",
            "casual": "'cause",
            "twitter": "bc",
        },
    ),
}

_IRREGULAR_VERBS: dict[str, dict[str, str]] = {
    "begin": {"past": "began", "gerund": "beginning"},
    "think": {"past": "thought"},
    "buy": {"past": "bought"},
    "tell": {"past": "told"},
    "grab": {"past": "grabbed", "gerund": "grabbing"},
    "reckon": {"past": "reckoned"},
}

_IRREGULAR_PLURALS: dict[str, str] = {"child": "children", "minor": "minors"}

# Words that typically precede a verb (used to disambiguate noun/verb homographs).
_BASE_VERB_CONTEXT = frozenset(
    {
        "i",
        "you",
        "we",
        "they",
        "to",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "please",
        "not",
        "don't",
        "didn't",
        "won't",
        "wouldn't",
        "can't",
        "couldn't",
        "shouldn't",
        "let's",
        "us",
        "always",
        "never",
        "often",
        "usually",
    }
)
_THIRD_PERSON_CONTEXT = frozenset({"he", "she", "it", "who", "this", "that", "which", "everyone", "someone"})

# --------------------------------------------------------------------------- #
# Phrase-level transforms
# --------------------------------------------------------------------------- #
Replacement = str | Callable[[random.Random], str]

_VERBOSE_PHRASES: list[tuple[str, str]] = [
    (r"\bin order to\b", "to"),
    (r"\bat this point in time\b", "now"),
    (r"\bat the present time\b", "now"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bin spite of the fact that\b", "although"),
    (r"\bin the event that\b", "if"),
    (r"\bwith regard to\b", "about"),
    (r"\bwith reference to\b", "about"),
    (r"\bwith respect to\b", "on"),
    (r"\bfor the purpose of\b", "for"),
    (r"\ba large number of\b", "many"),
    (r"\bon a daily basis\b", "daily"),
    (r"\bprior to\b", "before"),
    (r"\bsubsequent to\b", "after"),
    (r"\buntil such time as\b", "until"),
    (r"\bfor all intents and purposes\b", "essentially"),
    (r"\bit is important to note that\s*", ""),
    (r"\bthe fact of the matter is( that)?\s*", ""),
]

_FILLER_WORDS = r"\b(?:really|very|basically|actually|just|quite|simply|literally)\s+"

# "need to X" must become "must X", never "require to X".
_MODAL_NEED: list[tuple[str, Replacement]] = [
    (r"\b(?:need|needs) to\b", "must"),
    (r"\bneeded to\b", "had to"),
]

_STYLE_PHRASES: dict[str, list[tuple[str, Replacement]]] = {
    "legal": [
        *_MODAL_NEED,
        (r"\bmake sure\b", "ensure"),
        (r"\bpromise\b", "covenant"),
        (r"\bunderstand\b", "acknowledge"),
        (r"\bsend\b", "transmit"),
        (r"\bsay\b", "state"),
        (r"\bwhether or not\b", "whether"),
    ],
    "technical": [
        *_MODAL_NEED,
        (r"\blook at\b", "examine"),
        (r"\bfigure out\b", "determine"),
        (r"\bmake sure\b", "verify"),
        (r"\bset up\b", "configure"),
        (r"\bfind out\b", "determine"),
    ],
    "academic": [
        *_MODAL_NEED,
        (r"\bmake sure\b", "ensure"),
        (r"\bshow that\b", "demonstrate that"),
        (r"\ba lot of\b", "a considerable amount of"),
        (r"\blots of\b", "numerous"),
        (r"\bfind out\b", "ascertain"),
        (r"\blook into\b", "investigate"),
    ],
    "formal": [
        *_MODAL_NEED,
        (r"\bmake sure\b", "ensure"),
        (r"\ba lot of\b", "a great deal of"),
        (r"\blots of\b", "many"),
        (r"\bfind out\b", "determine"),
        (r"\bok(?:ay)?\b", "acceptable"),
        (r"\bthanks\b", "thank you"),
    ],
    "casual": [
        (r"\bhowever\b", "but"),
        (r"\btherefore\b", "so"),
        (r"\bpurchase\b", "buy"),
        (r"\bassist\b", "help"),
        (r"\brequire\b", "need"),
        (r"\bthank you\b", "thanks"),
    ],
    "creative": [
        (r"\bsaid\b", lambda rng: rng.choice(["exclaimed", "murmured", "declared"])),
        (r"\bwent\b", lambda rng: rng.choice(["ventured", "wandered", "journeyed"])),
        (r"\blooked\b", lambda rng: rng.choice(["gazed", "peered", "glanced"])),
        (r"\bwalked\b", lambda rng: rng.choice(["strolled", "ambled", "strode"])),
        (r"\bhappy\b", lambda rng: rng.choice(["joyful", "elated", "radiant"])),
        (r"\bsad\b", lambda rng: rng.choice(["sorrowful", "melancholy", "downcast"])),
    ],
    "twitter": [
        (r"\band\b", "&"),
        (r"\bwith\b", "w/"),
        (r"\bpeople\b", "ppl"),
    ],
}

# Contractions (expansion -> contraction). The look-ahead keeps clause-final
# forms intact ("that is what it is." must not become "...what it's.").
_CONTRACTIONS: dict[str, str] = {
    "do not": "don't",
    "does not": "doesn't",
    "did not": "didn't",
    "cannot": "can't",
    "can not": "can't",
    "will not": "won't",
    "would not": "wouldn't",
    "should not": "shouldn't",
    "could not": "couldn't",
    "is not": "isn't",
    "are not": "aren't",
    "was not": "wasn't",
    "were not": "weren't",
    "has not": "hasn't",
    "have not": "haven't",
    "had not": "hadn't",
    "i am": "I'm",
    "you are": "you're",
    "we are": "we're",
    "they are": "they're",
    "it is": "it's",
    "that is": "that's",
    "there is": "there's",
    "i will": "I'll",
    "you will": "you'll",
    "we will": "we'll",
    "they will": "they'll",
    "let us": "let's",
}
_EXPANSIONS: dict[str, str] = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "i'm": "I am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "i'll": "I will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "let's": "let us",
    "i've": "I have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
}
_CONTRACTING_STYLES = frozenset({"casual", "twitter", "concise"})
_EXPANDING_STYLES = frozenset({"formal", "legal", "academic", "technical"})


# --------------------------------------------------------------------------- #
# Inflection helpers
# --------------------------------------------------------------------------- #
def _is_vowel(char: str) -> bool:
    return char.lower() in "aeiou"


def inflect_verb(lemma: str, form: str) -> str:
    """Inflect the head word of ``lemma`` into ``form`` (base, third, past, gerund)."""
    head, sep, tail = lemma.partition(" ")
    if form == "base":
        return lemma
    irregular = _IRREGULAR_VERBS.get(head, {})
    if form in irregular:
        inflected = irregular[form]
    elif form == "third":
        if head.endswith(("s", "sh", "ch", "x", "z", "o")):
            inflected = head + "es"
        elif head.endswith("y") and len(head) > 1 and not _is_vowel(head[-2]):
            inflected = head[:-1] + "ies"
        else:
            inflected = head + "s"
    elif form == "past":
        if head.endswith("e"):
            inflected = head + "d"
        elif head.endswith("y") and len(head) > 1 and not _is_vowel(head[-2]):
            inflected = head[:-1] + "ied"
        else:
            inflected = head + "ed"
    elif form == "gerund":
        if head.endswith("ie"):
            inflected = head[:-2] + "ying"
        elif head.endswith("e") and not head.endswith("ee"):
            inflected = head[:-1] + "ing"
        else:
            inflected = head + "ing"
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown verb form: {form}")
    return inflected + sep + tail


def pluralize(noun: str) -> str:
    if noun in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[noun]
    if noun.endswith(("s", "sh", "ch", "x", "z")):
        return noun + "es"
    if noun.endswith("y") and len(noun) > 1 and not _is_vowel(noun[-2]):
        return noun[:-1] + "ies"
    return noun + "s"


def _match_case(original: str, replacement: str) -> str:
    if not replacement:
        return replacement
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _previous_word(text: str, index: int) -> str:
    match = re.search(r"([A-Za-z']+)[^A-Za-z']*$", text[:index])
    return match.group(1).lower() if match else ""


def _build_surface_forms() -> dict[str, tuple[str, str]]:
    """Map every surface form (e.g. ``used``) to ``(lemma, form)``."""
    forms: dict[str, tuple[str, str]] = {}
    for lemma, (pos, _ambiguous, _styles) in _LEXICON.items():
        if pos == "verb":
            for form in ("base", "third", "past", "gerund"):
                forms.setdefault(inflect_verb(lemma, form), (lemma, form))
        elif pos == "noun":
            forms.setdefault(lemma, (lemma, "singular"))
            forms.setdefault(pluralize(lemma), (lemma, "plural"))
        else:
            forms.setdefault(lemma, (lemma, "base"))
    return forms


_SURFACE_FORMS = _build_surface_forms()
_SURFACE_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _SURFACE_FORMS), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)


_CONTRACTION_PATTERNS = [
    (re.compile(r"\b" + re.escape(k) + r"\b(?=\s+[A-Za-z])", re.IGNORECASE), v)
    for k, v in sorted(_CONTRACTIONS.items(), key=lambda kv: len(kv[0]), reverse=True)
]
_EXPANSION_PATTERNS = [(_phrase_pattern(k), v) for k, v in _EXPANSIONS.items()]


# --------------------------------------------------------------------------- #
# Rule-based engine
# --------------------------------------------------------------------------- #
class RuleBasedRewriter:
    """Deterministic lexical style transfer (no network, no model downloads)."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def rewrite(self, text: str, style: str) -> str:
        if style not in STYLES:
            raise ValueError(f"Unsupported style: {style}")
        if not text.strip():
            return text

        result = text
        if style in ("concise", "twitter"):
            result = self._apply_patterns(result, [(p, r) for p, r in _VERBOSE_PHRASES])
            result = re.sub(_FILLER_WORDS, "", result, flags=re.IGNORECASE)
        result = self._apply_patterns(result, _STYLE_PHRASES.get(style, []))
        result = self._apply_lexicon(result, style)
        if style in _CONTRACTING_STYLES:
            result = self._apply_compiled(result, _CONTRACTION_PATTERNS)
        elif style in _EXPANDING_STYLES:
            result = self._apply_compiled(result, _EXPANSION_PATTERNS)
        result = self._tidy(result)
        if style == "twitter":
            result = truncate_text(result, TWITTER_LIMIT)
        return result

    # -- internals --------------------------------------------------------- #
    def _apply_patterns(self, text: str, rules: list[tuple[str, Replacement]]) -> str:
        for pattern, replacement in rules:
            if callable(replacement):
                text = re.sub(
                    pattern,
                    lambda m, fn=replacement: _match_case(m.group(0), fn(self._rng)),
                    text,
                    flags=re.IGNORECASE,
                )
            else:
                text = re.sub(
                    pattern,
                    lambda m, rep=replacement: _match_case(m.group(0), rep) if rep else "",
                    text,
                    flags=re.IGNORECASE,
                )
        return text

    @staticmethod
    def _apply_compiled(text: str, rules: list[tuple[re.Pattern[str], str]]) -> str:
        for pattern, replacement in rules:
            text = pattern.sub(lambda m, rep=replacement: _match_case(m.group(0), rep), text)
        return text

    def _apply_lexicon(self, text: str, style: str) -> str:
        def substitute(match: re.Match[str]) -> str:
            surface = match.group(0)
            lemma, form = _SURFACE_FORMS[surface.lower()]
            pos, ambiguous, styles = _LEXICON[lemma]
            replacement = styles.get(style)
            if not replacement or replacement == lemma:
                return surface
            if pos == "verb":
                if ambiguous and form in ("base", "third"):
                    previous = _previous_word(match.string, match.start())
                    allowed = _BASE_VERB_CONTEXT if form == "base" else _THIRD_PERSON_CONTEXT
                    if previous not in allowed:
                        return surface
                replacement = inflect_verb(replacement, form)
            elif pos == "noun" and form == "plural":
                replacement = pluralize(replacement)
            return _match_case(surface, replacement)

        return _SURFACE_PATTERN.sub(substitute, text)

    @staticmethod
    def _tidy(text: str) -> str:
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r",\s*,", ",", text)
        # Re-capitalise sentence starts that may have been removed or replaced.
        text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text.strip())
        return text


def truncate_text(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` characters, preferring sentence then word boundaries."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    sentence_end = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if window.endswith((".", "!", "?")):
        sentence_end = max(sentence_end, len(window) - 2)
    if sentence_end >= int(limit * 0.5):
        return window[: sentence_end + 1].rstrip()
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space >= int(limit * 0.5):
        cut = cut[:space]
    return cut.rstrip(" ,;:-") + "…"


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #
@dataclass
class RewriteResult:
    original: str
    rewritten: str
    style: str
    engine: str
    warnings: list[str] = field(default_factory=list)


_REWRITE_SYSTEM_PROMPT = (
    "You are a professional editor. You rewrite text in a requested style while "
    "preserving its meaning, facts, names, numbers and language. You never add new "
    "information and you reply with the rewritten text only."
)


class Ghostwriter:
    def __init__(self, llm: GeminiClient | None = None, rules: RuleBasedRewriter | None = None) -> None:
        self._llm = llm
        self._rules = rules or RuleBasedRewriter()

    def rewrite(self, text: str, style: str, engine: Engine = "auto") -> RewriteResult:
        if style not in STYLES:
            raise ValueError(f"Unsupported style: {style}")
        text = text.strip()
        warnings: list[str] = []
        rewritten: str | None = None
        engine_used = "rules"

        llm_ready = self._llm is not None and self._llm.is_configured
        if engine == "gemini" and not llm_ready:
            raise LLMNotConfiguredError()
        if engine in ("auto", "gemini") and llm_ready:
            assert self._llm is not None
            try:
                rewritten = self._llm.generate(
                    self._build_prompt(text, style),
                    system_instruction=_REWRITE_SYSTEM_PROMPT,
                    temperature=0.7,
                ).text
                rewritten = rewritten.strip().strip('"').strip()
                engine_used = "gemini"
            except LLMError as exc:
                if engine == "gemini":
                    raise
                logger.warning("Gemini rewrite failed, falling back to rules: %s", exc.message)
                warnings.append("Gemini is unavailable right now, so the offline rule-based engine was used.")

        if rewritten is None:
            rewritten = self._rules.rewrite(text, style)

        if style == "twitter" and len(rewritten) > TWITTER_LIMIT:
            rewritten = truncate_text(rewritten, TWITTER_LIMIT)
            warnings.append(f"Trimmed to fit the {TWITTER_LIMIT}-character limit.")
        if rewritten.strip().lower() == text.lower():
            warnings.append("No changes were needed for this style. Try another style or a longer passage.")

        return RewriteResult(original=text, rewritten=rewritten, style=style, engine=engine_used, warnings=warnings)

    @staticmethod
    def _build_prompt(text: str, style: str) -> str:
        return (
            f"Rewrite the text below in {STYLE_GUIDES[style]}.\n"
            "Keep the original meaning and language. Return only the rewritten text, "
            "without quotes, headings or explanations.\n\n"
            f'Text:\n"""\n{text}\n"""'
        )
