# 必要なライブラリのインポート
from deep_translator import GoogleTranslator
from transformers import pipeline

# 翻訳器の初期化（日本語から英語への翻訳）
translator = GoogleTranslator(source='ja', target='en')

# 感情分析モデルのロード
sentiment_model = pipeline("sentiment-analysis")

# 翻訳と感情分析を行う関数
def translate_and_analyze(text):
    """
    日本語のテキストを英語に翻訳し、感情分析を行う。
    :param text: 日本語の文章
    """
    # 日本語から英語に翻訳
    translated_text = translator.translate(text)
    print(f"翻訳されたテキスト: {translated_text}")

    # 翻訳されたテキストで感情分析を実行
    results = sentiment_model(translated_text)

    # 結果を表示
    for result in results:
        label = result['label']  # 'POSITIVE' or 'NEGATIVE'
        score = result['score']  # スコア
        print(f"感情分析結果 - ラベル: {label}, スコア: {score * 100}%")

if __name__ == "__main__":
    # テキストの入力
    input_text = input("分析したい日本語の文章を入力してください: ")

    # 翻訳と感情分析の実行
    translate_and_analyze(input_text)
