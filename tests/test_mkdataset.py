"""Tests for mkdataset.py"""

from pathlib import Path

import pytest
from PIL import Image

import mkdataset


@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary config.toml for testing."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
source_dir = "./my_source"
max_palette_colors = 16

image_extensions = [".png", ".jpg"]
skip_extensions = [".wav", ".mp3", ".map", ".pyc"]

[language_map]
".py" = "python"
".js" = "javascript"
".md" = "markdown"
".txt" = "text"
"""
    )
    return config_path


@pytest.fixture
def config(tmp_config):
    """Load config from temporary file."""
    return mkdataset.load_config(tmp_config)


@pytest.fixture
def tmp_source(tmp_path):
    """Create a temporary source directory with test files."""
    source = tmp_path / "source"
    proj = source / "test-project"
    proj.mkdir(parents=True)

    # Python file
    (proj / "hello.py").write_text("print('hello')\n")

    # JavaScript file
    (proj / "app.js").write_text("console.log('hi');\n")

    # Markdown file
    (proj / "README.md").write_text("# Test\n\nThis is a test.\n")

    # Text file with no extension
    (proj / "Makefile").write_text("all:\n\techo hello\n")

    # Small PNG image (2x2 RGBA)
    img = Image.new("RGBA", (2, 2), (255, 0, 0, 255))
    img.putpixel((1, 0), (0, 255, 0, 255))
    img.putpixel((0, 1), (0, 0, 255, 255))
    img.putpixel((1, 1), (255, 255, 0, 128))
    img.save(proj / "sprite.png")

    # Audio file (should be skipped)
    (proj / "sound.wav").write_bytes(b"\x00" * 100)

    # Binary file (no known extension, but has null bytes)
    (proj / "data.bin").write_bytes(b"\x00\x01\x02\x03\x00\xff")

    # Empty file (should be skipped)
    (proj / "empty.py").write_text("")

    return source


class TestLoadConfig:
    def test_load_config(self, tmp_config):
        config = mkdataset.load_config(tmp_config)
        assert ".png" in config["image_extensions"]
        assert ".wav" in config["skip_extensions"]
        assert config["language_map"][".py"] == "python"
        assert config["source_dir"] == "./my_source"

    def test_load_config_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            mkdataset.load_config(tmp_path / "nonexistent.toml")


class TestResolveSourceDir:
    def test_cli_overrides_config(self, config):
        result = mkdataset.resolve_source_dir("/cli/path", config)
        assert result == Path("/cli/path")

    def test_config_value_used(self, config):
        result = mkdataset.resolve_source_dir(None, config)
        assert result == Path("./my_source")

    def test_default_current_dir(self):
        result = mkdataset.resolve_source_dir(None, {})
        assert result == Path(".")


class TestIsBinaryContent:
    def test_text_file(self, tmp_path):
        f = tmp_path / "text.py"
        f.write_text("print('hello')\n")
        assert mkdataset.is_binary_content(f) is False

    def test_binary_file(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello\x00world")
        assert mkdataset.is_binary_content(f) is True

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty"
        f.write_bytes(b"")
        assert mkdataset.is_binary_content(f) is False


class TestReadFileSafe:
    def test_utf8(self, tmp_path):
        f = tmp_path / "utf8.py"
        f.write_text("# コメント\nprint('hello')\n", encoding="utf-8")
        content = mkdataset.read_file_safe(f)
        assert "コメント" in content
        assert "print('hello')" in content

    def test_latin1_fallback(self, tmp_path):
        f = tmp_path / "latin1.txt"
        f.write_bytes("caf\xe9".encode("latin-1"))
        content = mkdataset.read_file_safe(f)
        assert "caf" in content


class TestSanitizeVarname:
    def test_simple(self):
        assert mkdataset.sanitize_varname("sprite") == "sprite"

    def test_starts_with_digit(self):
        result = mkdataset.sanitize_varname("3d_model")
        assert result == "_3d_model"
        assert result.isidentifier()

    def test_special_chars(self):
        result = mkdataset.sanitize_varname("my-image (1)")
        assert result == "my_image__1_"
        assert result.isidentifier()


class TestGenerateUniqueVarnames:
    def test_all_unique_stems(self):
        paths = ["project/sprite.png", "project/icon.png", "project/bg.png"]
        result = mkdataset.generate_unique_varnames(paths)
        assert result["project/sprite.png"] == "sprite"
        assert result["project/icon.png"] == "icon"
        assert result["project/bg.png"] == "bg"

    def test_duplicate_stems_resolved_by_parent(self):
        paths = [
            "game/ghost/blinky/down1.png",
            "game/ghost/clyde/down1.png",
            "game/ghost/inky/down1.png",
        ]
        result = mkdataset.generate_unique_varnames(paths)
        assert result["game/ghost/blinky/down1.png"] == "blinky_down1"
        assert result["game/ghost/clyde/down1.png"] == "clyde_down1"
        assert result["game/ghost/inky/down1.png"] == "inky_down1"

    def test_needs_two_levels_to_resolve(self):
        paths = [
            "a/sub/img.png",
            "b/sub/img.png",
        ]
        result = mkdataset.generate_unique_varnames(paths)
        assert result["a/sub/img.png"] == "a_sub_img"
        assert result["b/sub/img.png"] == "b_sub_img"

    def test_mixed_unique_and_duplicate(self):
        paths = [
            "project/unique.png",
            "project/dir_a/icon.png",
            "project/dir_b/icon.png",
        ]
        result = mkdataset.generate_unique_varnames(paths)
        # unique stays short
        assert result["project/unique.png"] == "unique"
        # duplicates get parent prefix
        assert result["project/dir_a/icon.png"] == "dir_a_icon"
        assert result["project/dir_b/icon.png"] == "dir_b_icon"

    def test_digit_start_after_resolve(self):
        paths = [
            "player/1.png",
            "death/1.png",
        ]
        result = mkdataset.generate_unique_varnames(paths)
        for varname in result.values():
            assert varname.isidentifier()

    def test_single_path(self):
        result = mkdataset.generate_unique_varnames(["top/icon.png"])
        assert result["top/icon.png"] == "icon"

    def test_empty_list(self):
        result = mkdataset.generate_unique_varnames([])
        assert result == {}

    def test_top_level_files(self):
        paths = ["a.png", "b.png"]
        result = mkdataset.generate_unique_varnames(paths)
        assert result["a.png"] == "a"
        assert result["b.png"] == "b"


class TestFormatImageContent:
    def test_palette_format(self):
        palette = {0: [255, 0, 0, 255], 1: [0, 255, 0, 255]}
        pixels = [[0, 1], [1, 0]]
        varname = "sprite"
        image_info = {"width": 2, "height": 2, "channels": "RGBA", "num_colors": 2}
        result = mkdataset.format_image_content(
            palette, pixels, "project/sprite.png", varname, image_info
        )
        assert "# project/sprite.png (2x2, RGBA, 2 colors)" in result
        assert "sprite_palette = {0: [255, 0, 0, 255], 1: [0, 255, 0, 255]}" in result
        assert "sprite = [[0, 1], [1, 0]]" in result

    def test_comment_includes_color_count(self):
        palette = {0: [0, 0, 0, 0]}
        pixels = [[0]]
        image_info = {"width": 1, "height": 1, "channels": "RGBA", "num_colors": 1}
        result = mkdataset.format_image_content(
            palette, pixels, "icon.png", "icon", image_info
        )
        assert "1 colors" in result


class TestReadImageAsText:
    def test_basic_quantization(self, tmp_path):
        img = Image.new("RGBA", (2, 2), (255, 0, 0, 255))
        img.putpixel((1, 0), (0, 255, 0, 255))
        path = tmp_path / "test.png"
        img.save(path)

        palette, pixels, info = mkdataset.read_image_as_text(path, max_colors=16)
        assert info["width"] == 2
        assert info["height"] == 2
        assert info["channels"] == "RGBA"
        assert info["num_colors"] == 2
        assert len(palette) == 2
        assert 0 in palette and 1 in palette
        colors_set = {tuple(v) for v in palette.values()}
        assert (255, 0, 0, 255) in colors_set
        assert (0, 255, 0, 255) in colors_set
        assert len(pixels) == 2
        assert len(pixels[0]) == 2
        all_indices = {idx for row in pixels for idx in row}
        assert all_indices == set(palette.keys())

    def test_transparency_preserved(self, tmp_path):
        img = Image.new("RGBA", (2, 1), (255, 0, 0, 255))
        img.putpixel((1, 0), (0, 0, 0, 0))
        path = tmp_path / "test.png"
        img.save(path)

        palette, pixels, info = mkdataset.read_image_as_text(path, max_colors=16)
        has_transparent = any(v[3] == 0 for v in palette.values())
        assert has_transparent

    def test_max_colors_limits_palette(self, tmp_path):
        img = Image.new("RGBA", (16, 16))
        for x in range(16):
            for y in range(16):
                img.putpixel((x, y), (x * 16, y * 16, 128, 255))
        path = tmp_path / "test.png"
        img.save(path)

        palette, pixels, info = mkdataset.read_image_as_text(path, max_colors=4)
        assert len(palette) <= 4
        assert info["num_colors"] <= 4

    def test_fewer_colors_than_max(self, tmp_path):
        img = Image.new("RGBA", (2, 1), (100, 100, 100, 255))
        path = tmp_path / "test.png"
        img.save(path)

        palette, pixels, info = mkdataset.read_image_as_text(path, max_colors=256)
        assert len(palette) == 1
        assert info["num_colors"] == 1

    def test_indices_consecutive_from_zero(self, tmp_path):
        img = Image.new("RGBA", (3, 1), (255, 0, 0, 255))
        img.putpixel((1, 0), (0, 255, 0, 255))
        img.putpixel((2, 0), (0, 0, 255, 255))
        path = tmp_path / "test.png"
        img.save(path)

        palette, pixels, info = mkdataset.read_image_as_text(path, max_colors=16)
        assert sorted(palette.keys()) == list(range(len(palette)))

    def test_1x1_image(self, tmp_path):
        img = Image.new("RGBA", (1, 1), (42, 100, 200, 255))
        path = tmp_path / "test.png"
        img.save(path)

        palette, pixels, info = mkdataset.read_image_as_text(path, max_colors=16)
        assert len(palette) == 1
        assert pixels == [[0]]

    def test_jpeg_input(self, tmp_path):
        img = Image.new("RGB", (2, 2), (255, 0, 0))
        path = tmp_path / "test.jpg"
        img.save(path)

        palette, pixels, info = mkdataset.read_image_as_text(path, max_colors=16)
        assert info["channels"] == "RGBA"
        assert all(v[3] == 255 for v in palette.values())


class TestDetectLanguage:
    def test_known_extension(self, config):
        assert mkdataset.detect_language(Path("foo.py"), config) == "python"
        assert mkdataset.detect_language(Path("bar.js"), config) == "javascript"

    def test_image_extension(self, config):
        assert mkdataset.detect_language(Path("img.png"), config) == "image-data"
        assert mkdataset.detect_language(Path("photo.jpg"), config) == "image-data"

    def test_unknown_extension(self, config):
        assert mkdataset.detect_language(Path("file.xyz"), config) == "unknown"

    def test_no_extension(self, config):
        assert mkdataset.detect_language(Path("Makefile"), config) == "text"


class TestExtractProjectName:
    def test_nested_path(self):
        assert mkdataset.extract_project_name("project/src/main.py") == "project"

    def test_single_level(self):
        assert mkdataset.extract_project_name("project/file.py") == "project"

    def test_top_level_file(self):
        assert mkdataset.extract_project_name("file.py") == ""


class TestCountTokens:
    def test_without_tokenizer(self):
        # Fallback: len(text) // CHARS_PER_TOKEN
        assert mkdataset.count_tokens("abcdef", None) == 2  # 6 // 3
        assert mkdataset.count_tokens("ab", None) == 0  # 2 // 3

    def test_load_tokenizer_none(self):
        result = mkdataset.load_tokenizer({})
        assert result is None


class TestChunkText:
    def test_short_text_no_split(self):
        text = "line1\nline2\nline3"
        chunks = mkdataset.chunk_text(text, max_tokens=100, overlap_tokens=10)
        assert chunks == [text]

    def test_split_into_multiple_chunks(self):
        # CHARS_PER_TOKEN=3, so max_tokens=5 => max_chars=15
        lines = [f"line{i:02d}x" for i in range(10)]  # each "line00x" = 7 chars + newline
        text = "\n".join(lines)
        chunks = mkdataset.chunk_text(text, max_tokens=5, overlap_tokens=0)
        assert len(chunks) > 1
        # All original content should appear across chunks
        joined = "\n".join(chunks)
        for line in lines:
            assert line in joined

    def test_overlap_preserves_context(self):
        # CHARS_PER_TOKEN=3, max_tokens=10 => max_chars=30
        lines = ["aaaaaaaaaa"] * 6  # 10 chars each + newline = 11 per line
        text = "\n".join(lines)  # total ~65 chars
        chunks = mkdataset.chunk_text(text, max_tokens=10, overlap_tokens=4)
        assert len(chunks) >= 2
        # Overlapping lines should appear in consecutive chunks
        chunk0_lines = chunks[0].split("\n")
        chunk1_lines = chunks[1].split("\n")
        # Last line(s) of chunk 0 should appear at start of chunk 1
        assert chunk0_lines[-1] == chunk1_lines[0]

    def test_large_text_fits(self):
        text = "a" * 1000
        chunks = mkdataset.chunk_text(text, max_tokens=10000, overlap_tokens=0)
        assert chunks == [text]


class TestShouldSkipFile:
    def test_skip_audio(self, tmp_path, config):
        f = tmp_path / "sound.wav"
        f.write_bytes(b"\x00" * 100)
        assert mkdataset.should_skip_file(f, "", config, max_size=10_000_000) is True

    def test_skip_map(self, tmp_path, config):
        f = tmp_path / "bundle.map"
        f.write_text("{}")
        assert mkdataset.should_skip_file(f, "", config, max_size=10_000_000) is True

    def test_skip_large_file(self, tmp_path, config):
        f = tmp_path / "big.py"
        f.write_text("x" * 100)
        assert mkdataset.should_skip_file(f, "", config, max_size=50) is True

    def test_no_size_limit(self, tmp_path, config):
        f = tmp_path / "big.py"
        f.write_text("x" * 100)
        assert mkdataset.should_skip_file(f, "", config, max_size=None) is False

    def test_allow_python(self, tmp_path, config):
        f = tmp_path / "hello.py"
        f.write_text("print('hello')\n")
        assert mkdataset.should_skip_file(f, "", config, max_size=10_000_000) is False

    def test_allow_image(self, tmp_path, config):
        img = Image.new("RGBA", (2, 2), (255, 0, 0, 255))
        f = tmp_path / "sprite.png"
        img.save(f)
        assert mkdataset.should_skip_file(f, "", config, max_size=10_000_000) is False

    def test_skip_pattern_glob_recursive(self, tmp_path):
        """skip_patterns with ** matches nested paths."""
        config = {
            "image_extensions": [],
            "skip_extensions": [],
            "skip_patterns": ["scripts/tsgen/**"],
        }
        f = tmp_path / "scripts" / "tsgen" / "bin" / "Parser.js"
        f.parent.mkdir(parents=True)
        f.write_text("code")
        assert mkdataset.should_skip_file(f, "scripts/tsgen/bin/Parser.js", config, max_size=None) is True

    def test_skip_pattern_glob_wildcard(self, tmp_path):
        """skip_patterns with * matches filenames."""
        config = {
            "image_extensions": [],
            "skip_extensions": [],
            "skip_patterns": ["**/webpack*.config.js"],
        }
        f = tmp_path / "webpack.dist.config.js"
        f.write_text("module.exports = {}")
        assert mkdataset.should_skip_file(f, "plugins/spine/webpack.dist.config.js", config, max_size=None) is True

    def test_skip_pattern_no_match(self, tmp_path):
        """Files not matching skip_patterns are not skipped."""
        config = {
            "image_extensions": [],
            "skip_extensions": [],
            "skip_patterns": ["scripts/tsgen/**"],
        }
        f = tmp_path / "main.js"
        f.write_text("code")
        assert mkdataset.should_skip_file(f, "src/main.js", config, max_size=None) is False

    def test_skip_pattern_empty(self, tmp_path):
        """Empty skip_patterns skips nothing."""
        config = {
            "image_extensions": [],
            "skip_extensions": [],
            "skip_patterns": [],
        }
        f = tmp_path / "main.js"
        f.write_text("code")
        assert mkdataset.should_skip_file(f, "src/main.js", config, max_size=None) is False

    def test_skip_dts_extension(self, tmp_path):
        """'.d.ts' in skip_extensions is matched correctly."""
        config = {
            "image_extensions": [],
            "skip_extensions": [".d.ts"],
            "skip_patterns": [],
        }
        f = tmp_path / "spine-both.d.ts"
        f.write_text("declare module spine {}")
        assert mkdataset.should_skip_file(f, "runtime/spine-both.d.ts", config, max_size=None) is True


class TestBuildTextEntry:
    def test_text_entry_with_header(self):
        text = mkdataset.build_text_entry(
            content="print('hello')\n",
            rel_path="project/hello.py",
            language="python",
        )
        assert "### File: project/hello.py" in text
        assert "### Language: python" in text
        assert "print('hello')" in text

    def test_text_entry_no_header(self):
        text = mkdataset.build_text_entry(
            content="print('hello')\n",
            rel_path="project/hello.py",
            language="python",
            add_header=False,
        )
        assert "### File:" not in text
        assert text == "print('hello')\n"

    def test_image_entry_with_header(self):
        image_info = {"width": 16, "height": 16, "channels": "RGBA", "num_colors": 4}
        text = mkdataset.build_text_entry(
            content="sprite_palette = {0: [255, 0, 0, 255]}\nsprite = [[0]]",
            rel_path="project/sprite.png",
            language="image-data",
            image_info=image_info,
        )
        assert "### File: project/sprite.png" in text
        assert "### Language: image-data" in text
        assert "### Image: width=16, height=16, channels=RGBA, colors=4" in text
        assert "sprite_palette" in text


class TestCollectFiles:
    def test_skips_binary(self, tmp_source, config):
        files = mkdataset.collect_files(tmp_source, config, max_size=10_000_000)
        paths = [str(f) for f in files]
        assert not any("data.bin" in p for p in paths)

    def test_skips_audio(self, tmp_source, config):
        files = mkdataset.collect_files(tmp_source, config, max_size=10_000_000)
        paths = [str(f) for f in files]
        assert not any("sound.wav" in p for p in paths)

    def test_skips_empty(self, tmp_source, config):
        files = mkdataset.collect_files(tmp_source, config, max_size=10_000_000)
        paths = [str(f) for f in files]
        assert not any("empty.py" in p for p in paths)

    def test_includes_images(self, tmp_source, config):
        files = mkdataset.collect_files(tmp_source, config, max_size=10_000_000)
        paths = [str(f) for f in files]
        assert any("sprite.png" in p for p in paths)

    def test_includes_text_files(self, tmp_source, config):
        files = mkdataset.collect_files(tmp_source, config, max_size=10_000_000)
        paths = [str(f) for f in files]
        assert any("hello.py" in p for p in paths)
        assert any("app.js" in p for p in paths)
        assert any("README.md" in p for p in paths)


class TestBuildDataset:
    def test_build_dataset(self, tmp_source, config):
        files = mkdataset.collect_files(tmp_source, config, max_size=10_000_000)
        ds = mkdataset.build_dataset(files, tmp_source, config, add_header=True)

        assert len(ds) > 0
        assert "text" in ds.column_names
        assert "file_path" in ds.column_names
        assert "language" in ds.column_names
        assert "size_bytes" in ds.column_names
        assert "num_lines" in ds.column_names
        assert "num_tokens" in ds.column_names
        assert "project" in ds.column_names

        # num_tokens should be positive for all records
        assert all(t > 0 for t in ds["num_tokens"])

        # Check that image-data language exists
        languages = set(ds["language"])
        assert "image-data" in languages
        assert "python" in languages

        # Check project name
        projects = set(ds["project"])
        assert "test-project" in projects

    def test_build_dataset_with_chunking(self, tmp_path):
        """Large file should be split into multiple records."""
        source = tmp_path / "src"
        proj = source / "proj"
        proj.mkdir(parents=True)
        # Create a file large enough to be chunked (max_tokens=5 => 15 chars)
        (proj / "big.py").write_text("\n".join(f"line{i:03d}xx" for i in range(20)))

        config = {
            "image_extensions": [],
            "skip_extensions": [],
            "language_map": {".py": "python"},
            "max_tokens": 10,  # 30 chars max per chunk
            "chunk_overlap_tokens": 2,
        }
        files = mkdataset.collect_files(source, config, max_size=None)
        ds = mkdataset.build_dataset(files, source, config, add_header=False)

        assert len(ds) > 1
        # Chunked paths should have #chunk-N suffix
        chunked_paths = [p for p in ds["file_path"] if "#chunk-" in p]
        assert len(chunked_paths) == len(ds)

    def test_build_dataset_no_chunking_when_zero(self, tmp_path):
        """max_tokens=0 disables chunking."""
        source = tmp_path / "src"
        proj = source / "proj"
        proj.mkdir(parents=True)
        (proj / "big.py").write_text("x\n" * 1000)

        config = {
            "image_extensions": [],
            "skip_extensions": [],
            "language_map": {".py": "python"},
            "max_tokens": 0,
        }
        files = mkdataset.collect_files(source, config, max_size=None)
        ds = mkdataset.build_dataset(files, source, config, add_header=False)

        assert len(ds) == 1
        assert "#chunk-" not in ds["file_path"][0]


class TestWriteAndLoadDataset:
    def test_write_and_load(self, tmp_source, tmp_path, config):
        output_dir = tmp_path / "output"
        files = mkdataset.collect_files(tmp_source, config, max_size=10_000_000)
        ds = mkdataset.build_dataset(files, tmp_source, config, add_header=True)
        mkdataset.write_dataset(ds, output_dir)

        # Verify output files exist
        parquet_files = list(output_dir.glob("*.parquet"))
        assert len(parquet_files) > 0

        # Load dataset back
        from datasets import load_dataset

        loaded = load_dataset(str(output_dir), split="train")
        assert len(loaded) == len(ds)
        assert set(loaded.column_names) == set(ds.column_names)


class TestParseArgs:
    def test_defaults(self):
        args = mkdataset.parse_args([])
        assert args.source is None
        assert args.output == "./output"
        assert args.config == "./config.toml"
        assert args.max_file_size is None
        assert args.with_header is False
        assert args.dry_run is False

    def test_source_short(self):
        args = mkdataset.parse_args(["-s", "/data/src"])
        assert args.source == "/data/src"

    def test_source_long(self):
        args = mkdataset.parse_args(["--source", "/data/src"])
        assert args.source == "/data/src"

    def test_output_short(self):
        args = mkdataset.parse_args(["-o", "/data/out"])
        assert args.output == "/data/out"

    def test_output_long(self):
        args = mkdataset.parse_args(["--output", "/data/out"])
        assert args.output == "/data/out"

    def test_config_short(self):
        args = mkdataset.parse_args(["-c", "my.toml"])
        assert args.config == "my.toml"

    def test_config_long(self):
        args = mkdataset.parse_args(["--config", "my.toml"])
        assert args.config == "my.toml"

    def test_max_file_size(self):
        args = mkdataset.parse_args(["--max-file-size", "1024"])
        assert args.max_file_size == 1024

    def test_with_header(self):
        args = mkdataset.parse_args(["--with-header"])
        assert args.with_header is True

    def test_default_no_header(self):
        args = mkdataset.parse_args([])
        assert args.with_header is False

    def test_dry_run(self):
        args = mkdataset.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_all_options(self):
        args = mkdataset.parse_args([
            "-s", "/src",
            "-o", "/out",
            "-c", "c.toml",
            "--max-file-size", "500",
            "--with-header",
            "--dry-run",
        ])
        assert args.source == "/src"
        assert args.output == "/out"
        assert args.config == "c.toml"
        assert args.max_file_size == 500
        assert args.with_header is True
        assert args.dry_run is True


class TestDryRun:
    def test_dry_run(self, tmp_source, config, capsys):
        mkdataset.dry_run(tmp_source, config, max_size=10_000_000)
        captured = capsys.readouterr()
        assert "test-project" in captured.out
        assert "python" in captured.out or "file" in captured.out.lower()
        # Should show max tokens info
        assert "Max tokens:" in captured.out
        assert "tokens" in captured.out
