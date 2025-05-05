from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

# テキストからリレーションのリストを抽出するメソッド (model)
def extract_relations_from_model_output(text):
    relations = []  # 抽出したリレーションを格納するためのリストを初期化
    subject, relation, object = "", "", ""  # リレーションの各要素 (主語, 関係, 目的語) を格納するための変数を初期化
    text = text.strip() # テキスト前後の空白文字を削除
    current = "x"   # 現在の解析対象を示すための変数を "x" で初期化
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():  # テキストから特殊トークン (<s>, <pad>, </s>) を削除し, 空白で分割してトークンのリストを作成
        if token == "<triplet>":    # 現在のトークンが <triplet> の場合
            current = "t"   # 現在の解析対象を示すための変数を "t" に設定
            if relation != "":  # relation が空でない場合
                relations.append({"head":subject.strip(), "type":relation.strip(), "tail":object.strip()})  # subject, relation, object を relations に追加
                relation = ""   # relation をリセット
            subject = ""    # subject をリセット
        elif token == "<subj>": # 現在のトークンが <subj> の場合
            current = "s"   # 現在の解析対象を示すための変数を "s" に設定
            if relation != "":  # relation が空でない場合
                relations.append({"head":subject.strip(), "type":relation.strip(), "tail":object.strip()})  # subject, relation, object を relations に追加
            object = "" # object をリセット
        elif token == "<obj>":  # 現在のトークンが <obj> の場合
            current = "o"   # 現在の解析対象を示すための変数を "o" に設定
            relation = ""   # relation をリセット
        else:   # 特殊トークンでない場合 (特殊トークンに対応するテキストを追加する処理)
            if current == "t":  # 現在の解析対象を示すための変数が "t" の場合
                subject += " " + token  # subject にトークンを追加
            elif current == "s":    # 現在の解析対象を示すための変数が "s" の場合
                object += " " + token   # object にトークンを追加
            elif current == "o":    # 現在の解析対象を示すための変数が "o" の場合
                relation += " " + token # relation にトークンを追加
    if subject != "" and relation != "" and object != "":   # ループ終了後に, subject, relation, object がすべて空でない場合
        relations.append({"head":subject.strip(), "type":relation.strip(), "tail":object.strip()})  # subject, relation, object を relations に追加
    return relations    # relations を返却

# 正解数をカウントするメソッド
def match_triplet_count(correct_triplets, generated_triplets):
    match_triplets_count = 0    # 正解数を初期化
    for generated_triplet in generated_triplets:    # 生成されたリレーションを順に代入
        if generated_triplet in correct_triplets:   # 生成されたリレーションがサンプルリレーションと完全一致している場合
            match_triplets_count += 1   # インクリメント
    return match_triplets_count # 正解数を返却

model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large") # モデルのロード
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large") # トークナイザーのロード
total_match_triplets_count = 0  # 総正解数を初期化
total_generated_triplets_count = 0  # 総生成数を初期化

test_data = load_dataset("Babelscape/rebel-dataset", split = "test[:100]")  # データセットのロード
for data in test_data:  # 各サンプルを順に読み込み
    correct_triplets = extract_relations_from_model_output(data["triplets"])    # テキストからリレーションのリストを抽出
    num_beams = len(correct_triplets)   # ビームサーチの幅をサンプルリレーションと同数に設定
    num_return_sequences = len(correct_triplets)    # 生成されるシーケンス数をサンプルリレーションと同数に設定
    model_inputs = tokenizer(data["context"], max_length = 256, padding = True, truncation = True, return_tensors = 'pt')   # テキストをトークン化
    gen_kwargs = {"max_length":256, "num_beams":num_beams, "num_return_sequences":num_return_sequences} # テキスト生成のパラメータを設定
    generated_tokens = model.generate(model_inputs["input_ids"].to(model.device), attention_mask = model_inputs["attention_mask"].to(model.device), **gen_kwargs)   # 設定したパラメータを用いて, モデルから生成されたトークンを取得
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens = False)   # 生成されたトークンをデコードして, 生成テキスト (特殊トークン (<triplet>, <subj>, <relation>, <obj>) と対応するトークンを含む形) のリストを作成
    generated_triplets = extract_relations_from_model_output(decoded_preds[0])  # テキストからリレーションのリストを抽出
    generated_triplets_count = len(generated_triplets)  # 生成されたリレーションの数を取得
    total_generated_triplets_count += generated_triplets_count  # 総生成数に追加
    match_triplets_count = match_triplet_count(correct_triplets, generated_triplets)    # 正解数を取得
    total_match_triplets_count += match_triplets_count  # 総正解数に追加

Accuracy = total_match_triplets_count / total_generated_triplets_count  # 正解率 (Accuracy) を計算
print(f"total_match_triplets_count:{total_match_triplets_count}")   # 総正解数を表示
print(f"total_generated_triplets_count:{total_generated_triplets_count}")   # 総生成数を表示
print(f"正解率 (Accuracy):{Accuracy}")  # 正解率 (Accuracy) を表示
