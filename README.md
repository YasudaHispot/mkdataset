# mkdataset

ソースコードファイルから、LLMの継続事前学習（CPT: Continued Pre-Training）用 HuggingFace データセットを生成する CLI ツール。

## 特徴

- ソースコードをテキストとして読み込み、Parquet 形式のデータセットに変換
- 画像ファイル（`.png`, `.jpg` 等）はピクセル配列をテキスト化して学習データに含める
- 音声・動画・バイナリなど学習に不要なファイルは自動除外
- モデルのトークナイザーを使った正確なトークン数カウントとチャンク分割
- `max_tokens` を超えるファイルは行単位で自動分割（オーバーラップ付き）
- 拡張子ごとの言語判定、除外ルール、画像判定はすべて `config.toml` で設定可能
- 出力は `load_dataset("./output", split="train")` でそのまま読み込み可能

## 必要環境

- Python 3.12 以上
- [uv](https://docs.astral.sh/uv/) パッケージマネージャ

## セットアップ

```bash
# リポジトリをクローン後、依存パッケージをインストール
uv sync
```

## ディレクトリ構成

```
mkdataset/
├── mkdataset.py          # CLI ツール本体
├── config.toml            # 設定ファイル
├── pyproject.toml         # プロジェクト設定
├── CLAUDE.md              # Claude Code 用プロジェクトルール
├── tests/
│   └── test_mkdataset.py  # テストコード
├── source/                # データセット元ファイル（ユーザーが配置）
│   ├── ProjectA/
│   └── ProjectB/
└── output/                # 生成されたデータセット
    └── train-00000-of-00001.parquet
```

## 使い方

### 基本的な実行

ソースディレクトリにソースコードを配置し、以下を実行します。

```bash
uv run mkdataset
```

`output/` ディレクトリに Parquet ファイルが生成されます。

### ドライラン（事前確認）

実際にデータセットを生成せず、対象ファイル一覧と統計情報（トークン数含む）を確認できます。

```bash
uv run mkdataset --dry-run
```

出力例:

```
Tokenizer: unsloth/gpt-oss-20b
  PacMan-main/README.md  (markdown, 1,234 bytes, 312 tokens)
  PacMan-main/pacman.py  (python, 5,678 bytes, 1,523 tokens)
  PacMan-main/assets/ghost_images/blinky/down1.png  (image-data, 892 bytes, 9,447 tokens)
  ...

--- Statistics ---
Total files: 2261
Total size: 9,083,580 bytes (8.7 MB)

Max tokens: 28,634 (phaser-3.90.0/src/gameobjects/particles/ParticleEmitter.js)
Tokenizer: unsloth/gpt-oss-20b

By language:
  javascript: 2126
  image-data: 55
  ...

By project:
  phaser-3.90.0: 2185
  PacMan-main: 76
```

### CLI オプション一覧

| オプション | 短縮形 | デフォルト値 | 説明 |
|-----------|-------|------------|------|
| `--source` | `-s` | config `source_dir` または `.` | ソースディレクトリのパス |
| `--output` | `-o` | `./output` | 出力ディレクトリのパス |
| `--config` | `-c` | `./config.toml` | 設定ファイルのパス |
| `--max-file-size` | | 無制限 | 最大ファイルサイズ（バイト単位） |
| `--no-header` | | ― | text カラムにヘッダーを付けない |
| `--dry-run` | | ― | 出力せずファイル一覧と統計を表示 |

### 使用例

```bash
# 別のソースディレクトリから生成
uv run mkdataset -s ./my_code -o ./my_dataset

# ファイルサイズ上限を 5MB に変更
uv run mkdataset --max-file-size 5242880

# ヘッダーなしで生成（ファイル内容のみ）
uv run mkdataset --no-header

# 別の設定ファイルを使用
uv run mkdataset -c ./custom_config.toml
```

## 設定ファイル（config.toml）

### source_dir — ソースディレクトリ

CLI `--source` で上書き可能。デフォルトはカレントディレクトリ。

```toml
source_dir = "./source"
```

### tokenizer — トークナイザー

HuggingFace モデル名を指定すると、正確なトークン数でチャンク分割・統計表示を行います。未指定の場合は概算（1トークン ≒ 3文字）で処理します。

```toml
tokenizer = "unsloth/gpt-oss-20b"
```

使用するモデルの `max_position_embeddings`（コンテキスト長）は、HuggingFace のモデルページの `config.json` で確認できます。

### max_tokens / chunk_overlap_tokens — チャンク分割

`max_tokens` を超えるテキストファイルは行単位で分割されます。`chunk_overlap_tokens` はチャンク間の重複トークン数で、分割箇所の文脈を維持します。

```toml
max_tokens = 8192          # チャンクあたりの最大トークン数（0 = 分割しない）
chunk_overlap_tokens = 256 # チャンク間のオーバーラップトークン数
```

`max_tokens` はモデルの `max_seq_length` に合わせて設定してください。

### image_extensions — 画像としてテキスト変換する拡張子

該当する拡張子のファイルは PIL で読み込み、ピクセル配列を Python リスト表現に変換して `text` カラムに格納します。

```toml
image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"]
```

### skip_extensions — 除外する拡張子

該当する拡張子のファイルはデータセットに含めません。

```toml
skip_extensions = [
    ".wav", ".mp3", ".ogg", ".mp4", ".webm",       # 音声/動画
    ".map",                                           # ソースマップ
    ".zip", ".tar", ".gz", ".bz2",                   # アーカイブ
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".wasm",  # コンパイル済み
    ".ttf", ".woff", ".woff2", ".eot",               # フォント
    ".ico", ".svg",                                   # アイコン/ベクター画像
]
```

### language_map — 拡張子から言語へのマッピング

`text` カラムのヘッダーと `language` カラムに使用されます。

- ここにない拡張子は `"unknown"` になります
- 拡張子なしのファイルは `"text"` になります
- 画像拡張子は自動的に `"image-data"` になるため、ここに書く必要はありません

```toml
[language_map]
".py" = "python"
".js" = "javascript"
".ts" = "typescript"
".json" = "json"
".md" = "markdown"
".txt" = "text"
".frag" = "glsl"
".vert" = "glsl"
```

## 出力データセットの構造

### スキーマ（7 カラム）

| カラム | 型 | 説明 |
|--------|------|------|
| `text` | string | ヘッダー + ファイル内容（LLM が学習する本文） |
| `file_path` | string | source/ からの相対パス（チャンク分割時は `#chunk-N` 付き） |
| `language` | string | 検出言語（`python`, `javascript`, `image-data` 等） |
| `size_bytes` | int64 | 元ファイルのサイズ（バイト） |
| `num_lines` | int64 | レコードの行数 |
| `num_tokens` | int64 | レコードのトークン数 |
| `project` | string | トップレベルディレクトリ名 |

### text カラムのフォーマット

テキストファイルの場合:

```
### File: PacMan-main/pacman.py
### Language: python

import sys
from model.board_definition import BoardDefinition
...
```

画像ファイルの場合:

```
### File: PacMan-main/assets/ghost_images/blinky/down1.png
### Language: image-data
### Image: width=28, height=28, channels=RGBA

[[[255, 0, 0, 255], [255, 0, 0, 255], ...], ...]
```

チャンク分割されたファイルの場合:

```
### File: phaser-3.90.0/src/input/InputPlugin.js#chunk-2
### Language: javascript

... (前のチャンクとオーバーラップあり)
```

### データセットの読み込み

```python
from datasets import load_dataset

ds = load_dataset("./output", split="train")
print(ds)
# Dataset({
#     features: ['text', 'file_path', 'language', 'size_bytes', 'num_lines', 'num_tokens', 'project'],
#     num_rows: 2326
# })

# 最初のレコードを確認
print(ds[0]["text"][:200])
print(ds[0]["file_path"])
print(ds[0]["num_tokens"])

# 言語ごとのレコード数
from collections import Counter
print(Counter(ds["language"]))

# 画像データのみ抽出
image_rows = ds.filter(lambda x: x["language"] == "image-data")
print(f"画像データ: {len(image_rows)} 件")

# 最大トークン数のレコード
max_idx = ds["num_tokens"].index(max(ds["num_tokens"]))
print(f"Max tokens: {ds[max_idx]['num_tokens']} ({ds[max_idx]['file_path']})")
```

## フィルタリングの仕組み

ファイルは以下の順序でフィルタリングされます。

1. **除外拡張子チェック** — `skip_extensions` に該当するファイルはスキップ
2. **ファイルサイズチェック** — `--max-file-size` を超えるファイルはスキップ（デフォルト無制限）
3. **画像拡張子チェック** — `image_extensions` に該当するファイルはピクセル配列テキストとして処理
4. **バイナリ検出** — 先頭 8KB にヌルバイトを含むファイルはスキップ（設定にない未知バイナリ対策）
5. **空ファイル** — 内容が空のテキストファイルはスキップ

## 開発

### テスト実行

```bash
uv run pytest
```

### カバレッジ付きテスト

```bash
uv run pytest --cov=mkdataset --cov-report=term-missing
```

### Lint / フォーマット

```bash
uv run ruff check .        # lint チェック
uv run ruff check --fix .  # lint 自動修正
uv run ruff format .       # コードフォーマット
```

## ライセンス

MIT
