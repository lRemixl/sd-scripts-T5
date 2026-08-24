import ast
import math
import os
import random
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
TRAIN_UTIL_PATH = ROOT / "library" / "train_util.py"

MD_FORMATS = ("_min_individual_md", "_character_thoughts", "_comic_md", "_long_thoughts")
JSON_FORMATS = ("_min_individual_json", "_json_names", "_json_no_names")
TEXT_FORMATS = ("_short", "_long_names", "_long_no_names")
ALL_FORMATS = MD_FORMATS + JSON_FORMATS + TEXT_FORMATS


class _Logger:
    def info(self, *args, **kwargs):
        pass


def _load_caption_contract():
    tree = ast.parse(TRAIN_UTIL_PATH.read_text(encoding="utf-8-sig"), filename=str(TRAIN_UTIL_PATH))
    helper_names = {
        "natural_language_caption_format",
        "is_natural_language_caption_path",
        "is_artist_style_tag",
        "_replace_natural_language_wildcards",
        "_extract_natural_language_tags",
        "_shuffle_natural_language_sentences",
        "_shuffle_natural_language_lists_and_paragraphs",
        "_shuffle_natural_language_h2",
        "_shuffle_natural_language_markdown",
        "_shuffle_natural_language_json",
        "process_natural_language_caption",
        "latent_cache_batch_sort_key",
    }
    helpers = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    dataset_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseDataset")
    process_caption = next(
        node for node in dataset_class.body if isinstance(node, ast.FunctionDef) and node.name == "process_caption"
    )
    module = ast.Module(body=[*helpers, process_caption], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "BaseSubset": object,
        "NATURAL_LANGUAGE_CAPTION_MD_FORMATS": MD_FORMATS,
        "NATURAL_LANGUAGE_CAPTION_JSON_FORMATS": JSON_FORMATS,
        "NATURAL_LANGUAGE_CAPTION_TEXT_FORMATS": TEXT_FORMATS,
        "NATURAL_LANGUAGE_CAPTION_FORMATS": ALL_FORMATS,
        "ARTIST_TAG_PREFIXES": ("@ ", "by "),
        "Optional": Optional,
        "Tuple": Tuple,
        "logger": _Logger(),
        "math": math,
        "os": os,
        "random": random,
        "re": re,
    }
    exec(compile(module, str(TRAIN_UTIL_PATH), "exec"), namespace)
    return namespace


def _subset(**overrides):
    values = {
        "caption_prefix": None,
        "caption_suffix": None,
        "caption_dropout_rate": 0.0,
        "caption_dropout_every_n_epochs": 0,
        "caption_tag_dropout_rate": 0.0,
        "artist_tag_dropout_rate": 0.0,
        "enable_wildcard": False,
        "shuffle_caption": False,
        "token_warmup_step": 0,
        "token_warmup_min": 1,
        "caption_separator": ",",
        "keep_tokens": 0,
        "keep_tokens_separator": None,
        "secondary_separator": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _dataset():
    return SimpleNamespace(
        current_epoch=1,
        current_step=0,
        max_train_steps=100,
        log_caption_dropout=False,
        log_caption_tag_dropout=False,
        replacements={},
    )


def test_all_natural_language_filename_formats_are_detected_exactly():
    contract = _load_caption_contract()
    detect = contract["natural_language_caption_format"]
    for caption_format in ALL_FORMATS:
        assert detect(f"sample{caption_format}.webp") == caption_format
    assert detect("sample_shorter.webp") is None
    assert detect("sample.webp") is None


def test_natural_language_caption_preserves_multiline_prose_while_generic_caption_does_not():
    contract = _load_caption_contract()
    process = contract["process_caption"]
    caption = "First sentence.\nSecond sentence."
    natural = process(_dataset(), _subset(), caption, "sample_short.webp")
    generic = process(_dataset(), _subset(), caption, "sample.webp")
    assert "First sentence." in natural
    assert "Second sentence." in natural
    assert generic == "First sentence."


def test_natural_language_shuffle_preserves_prose_and_trailing_tags():
    contract = _load_caption_contract()
    process = contract["process_caption"]
    caption = "First sentence. Second sentence. Third sentence. by old artist, newest"
    random.seed(4)
    result = process(_dataset(), _subset(shuffle_caption=True), caption, "sample_short.webp")
    for sentence in ("First sentence.", "Second sentence.", "Third sentence."):
        assert sentence in result
    assert "by old artist" in result
    assert "newest" in result
    assert result != caption


def test_natural_language_marker_detection_does_not_split_words_containing_old():
    contract = _load_caption_contract()
    process = contract["process_caption"]
    caption = "A golden-hour scene. Another descriptive sentence."
    result = process(_dataset(), _subset(), caption, "sample_short.webp")
    assert "golden-hour" in result
    assert "Another descriptive sentence." in result


def test_natural_language_wildcards_keep_the_rest_of_the_multiline_caption():
    contract = _load_caption_contract()
    process = contract["process_caption"]
    random.seed(2)
    caption = "A {red|blue} coat.\nA second sentence."
    result = process(_dataset(), _subset(enable_wildcard=True), caption, "sample_long_names.webp")
    assert ("red coat" in result) != ("blue coat" in result)
    assert "A second sentence." in result


def test_structured_natural_language_formats_keep_their_fields_and_headings():
    contract = _load_caption_contract()
    process = contract["process_caption"]
    markdown = process(
        _dataset(),
        _subset(),
        "# Scene\nA detailed setting.\n\n## Subject\nA central figure. by artist, newest",
        "sample_min_individual_md.webp",
    )
    structured_json = process(
        _dataset(),
        _subset(),
        "scene: A detailed setting.\nsubject: A central figure. by artist, newest",
        "sample_min_individual_json.webp",
    )
    assert "# Scene" in markdown
    assert "## Subject" in markdown
    assert "by artist, newest" in markdown
    assert "scene:" in structured_json
    assert "subject:" in structured_json
    assert "by artist, newest" in structured_json


def test_artist_dropout_exclusively_controls_artist_tags_when_enabled(monkeypatch):
    contract = _load_caption_contract()
    process = contract["process_caption"]
    monkeypatch.setattr(random, "random", lambda: 0.75)

    result = process(
        _dataset(),
        _subset(caption_tag_dropout_rate=1.0, artist_tag_dropout_rate=0.5),
        "ordinary tag, @ Artist Name, by Painter Name",
        "sample.webp",
    )

    assert result == "@ Artist Name, by Painter Name"


def test_normal_tag_dropout_still_controls_artist_tags_without_artist_rate(monkeypatch):
    contract = _load_caption_contract()
    process = contract["process_caption"]
    monkeypatch.setattr(random, "random", lambda: 0.75)

    result = process(
        _dataset(),
        _subset(caption_tag_dropout_rate=1.0),
        "ordinary tag, @ Artist Name, by Painter Name",
        "sample.webp",
    )

    assert result == ""


def test_artist_dropout_applies_to_fixed_and_flexible_caption_tags(monkeypatch):
    contract = _load_caption_contract()
    process = contract["process_caption"]
    monkeypatch.setattr(random, "random", lambda: 0.0)

    result = process(
        _dataset(),
        _subset(keep_tokens=1, artist_tag_dropout_rate=1.0),
        "@ Fixed Artist, ordinary tag, by Flexible Artist",
        "sample.webp",
    )

    assert result == "ordinary tag"


def test_latent_cache_sort_groups_equal_area_conditions_into_full_batches():
    contract = _load_caption_contract()
    sort_key = contract["latent_cache_batch_sort_key"]
    portrait = SimpleNamespace(bucket_reso=(384, 640))
    landscape = SimpleNamespace(bucket_reso=(640, 384))
    normal = SimpleNamespace(flip_aug=False, alpha_mask=False, random_crop=False)
    flipped = SimpleNamespace(flip_aug=True, alpha_mask=False, random_crop=False)
    records = [
        (portrait, normal),
        (landscape, normal),
        (portrait, flipped),
        (landscape, normal),
        (portrait, normal),
        (portrait, flipped),
    ]
    records.sort(key=lambda item: sort_key(*item))
    conditions = [
        (info.bucket_reso, subset.flip_aug, subset.alpha_mask, subset.random_crop)
        for info, subset in records
    ]
    runs = []
    for condition in conditions:
        if not runs or runs[-1][0] != condition:
            runs.append([condition, 0])
        runs[-1][1] += 1
    assert len(runs) == 3
    assert sorted(size for _, size in runs) == [2, 2, 2]


def test_cache_latents_uses_the_same_full_condition_sort_key_as_its_flush_logic():
    source = TRAIN_UTIL_PATH.read_text(encoding="utf-8-sig")
    assert "key=lambda info: latent_cache_batch_sort_key(" in source
    assert "self.image_to_subset[info.image_key]" in source
    assert "current_condition != condition" in source
    assert "self.process_caption(subset, image_info.caption, image_info.absolute_path)" in source

    custom_sdxl_source = (ROOT / "library" / "custom_sdxl_utils.py").read_text(encoding="utf-8-sig")
    assert "train_util.latent_cache_batch_sort_key(" in custom_sdxl_source
    assert "image_to_subset[info.image_key]" in custom_sdxl_source
