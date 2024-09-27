# 日本語感情分析ツール

このプロジェクトは、日本語のテキストを英語に翻訳し、その感情を分析するPythonプログラムです。深層学習モデルを使用して感情をポジティブまたはネガティブとして分類します。

## 機能

- 日本語のテキストを英語に翻訳
- 翻訳されたテキストの感情分析
- 感情分析の結果をスコア付きで表示

## 使用しているモデル

1. **翻訳モデル**:
   - **Google翻訳**:
     - `deep-translator`ライブラリを使用してGoogle翻訳APIにアクセスし、日本語から英語への翻訳を行います。

2. **感情分析モデル**:
   - **Hugging Face Transformers**:
     - このプログラムでは、Hugging FaceのTransformersライブラリから事前学習済みの感情分析モデルを使用しています。具体的には、以下のモデルを利用しています。

   - **モデル名**: `distilbert-base-uncased-finetuned-sst-2-english`
     - このモデルは、DistilBERTアーキテクチャに基づいており、英語の感情分析タスクのために微調整されています。
     - DistilBERTは、BERTの軽量版で、同等の性能を維持しつつ、計算資源と推論速度を大幅に削減しています。
     - このモデルは、ポジティブとネガティブの感情を分類するために訓練されています。

### 感情分析の詳細

- **感情分類**:
  - プログラムは翻訳されたテキストに基づいて、ポジティブまたはネガティブのラベルを付与します。

- **スコアの解釈**:
  - スコアはモデルがその感情が正しいと考える確率を示します。たとえば、スコアが98.75%の場合、モデルはそのテキストがポジティブである可能性が98.75%と判断します。

- **適用例**:
  - 顧客フィードバックの評価、ソーシャルメディアの投稿の分析、レビューの感情評価など、さまざまな領域で利用できます。

## 必要条件

以下のPythonライブラリが必要です。

- `transformers`: Hugging Faceの自然言語処理モデルを使用するためのライブラリ。
- `torch`: PyTorchは深層学習モデルの訓練および推論を行うためのフレームワーク。
- `deep-translator`: 様々な翻訳APIにアクセスするためのライブラリ。

## インストール

以下のコマンドを使用して必要なライブラリをインストールします。

```bash
pip install transformers torch deep-translator
