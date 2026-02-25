# mkdataset

LLM CPT（継続事前学習）用データセットビルダー。

## 開発環境

- パッケージ管理: uv
- Python仮想環境: uv管理

## コマンド

### セットアップ
uv sync

### テスト実行
uv run pytest

### カバレッジ付きテスト
uv run pytest --cov=mkdataset --cov-report=term-missing

### Lint チェック
uv run ruff check .

### フォーマット
uv run ruff format .

### Lint 自動修正
uv run ruff check --fix .

### データセット生成
uv run python mkdataset.py

### ドライラン（生成せず一覧表示）
uv run python mkdataset.py --dry-run

## 開発方針

- TDD（テスト駆動開発）: テストを先に書いてから実装する
- 単一ファイル構成: mkdataset.py に全ロジックを集約
- 設定はconfig.tomlで管理（画像拡張子、除外拡張子、言語マッピング）
- コード変更後は必ず lint と test を実行する

## プロジェクト構造

- mkdataset.py: CLIツール本体
- config.toml: 設定ファイル
- tests/test_mkdataset.py: テストコード
- source/: データセット元ファイル（git管理外）
- output/: 生成されたデータセット（git管理外）
