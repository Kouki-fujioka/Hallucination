import wikipedia
import pandas as pd
import torch
import math
from sentence_transformers import util, SentenceTransformer
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from neo4j import GraphDatabase

# Knoeledge Base (リレーション管理)
class KB():
    def __init__(self):
        self.relations = [] # クラスのインスタンスが作成された場合に relations リストを初期化

    def get_wikipedia_data(self, entity):
        try:    # wikipedia に entity のページが存在する場合
            page = wikipedia.page(entity, auto_suggest = False)   # entity のページ情報を取得
            entity_data = {"title":page.title} # entity のページタイトルを取得
            return entity_data  # entity_data を返却
        except: # wikipedia に entity のページが存在しない場合
            entity_data = {"title":entity} # entity をタイトルに設定
            return entity_data  # entity_data を返却

    def are_relations_equal(self, r1, r2):
        return all(r1[attribute] == r2[attribute] for attribute in ["head", "type", "tail"])   # r1 と r2 の各属性 (head, type, tail) が全て一致する場合に True を返却

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)   # r1 が既に relations リストに存在するかどうかを確認し, 既存の関係と等しい関係が 1 つでもあれば True を返却

    def add_relation(self, r):
        entities = [r["head"], r["tail"]]   # エンティティを取得
        entities_data = [self.get_wikipedia_data(entity) for entity in entities]  # エンティティのページ情報を取得
        r["head"] = entities_data[0]["title"]   # エンティティのリネーム (正規化)
        r["tail"] = entities_data[1]["title"]   # エンティティのリネーム (正規化)
        if not self.exists_relation(r): # r が relations リストに存在しない場合
            self.relations.append(r)    # r を relations リストに追加

    def print(self):
        # pd.set_option('display.max_rows', None) # 行の最大表示数を無制限に設定
        # pd.set_option('display.max_columns', None)  # 列の最大表示数を無制限に設定
        # df = pd.DataFrame(self.relations)   # relations リストに含まれるすべてのリレーションを pandas の DataFrame に変換
        # print(df)   # DataFrame を表示
        for relation in self.relations: # リレーションを順に代入
            print(relation) # リレーションを表示

# テキストからリレーションのリストを抽出するメソッド (model)
def extract_relations_from_model_output(text, label):
    relations = []  # 抽出したリレーションを格納するためのリストを初期化
    subject, relation, object = "", "", ""  # リレーションの各要素 (主語, 関係, 目的語) を格納するための変数を初期化
    text = text.strip() # テキスト前後の空白文字を削除
    current = "x"   # 現在の解析対象を示すための変数を "x" で初期化
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():  # テキストから特殊トークン (<s>, <pad>, </s>) を削除し, 空白で分割してトークンのリストを作成
        if token == "<triplet>":    # 現在のトークンが <triplet> の場合
            current = "t"   # 現在の解析対象を示すための変数を "t" に設定
            if relation != "":  # relation が空でない場合
                relations.append({"head":subject.strip(), "type":relation.strip(), "tail":object.strip(), "label":label})   # subject, relation, object, label を relations に追加
                relation = ""   # relation をリセット
            subject = ""    # subject をリセット
        elif token == "<subj>": # 現在のトークンが <subj> の場合
            current = "s"   # 現在の解析対象を示すための変数を "s" に設定
            if relation != "":  # relation が空でない場合
                relations.append({"head":subject.strip(), "type":relation.strip(), "tail":object.strip(), "label":label})   # subject, relation, object, label を relations に追加
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
    if subject != "" and relation != "" and object != "" and label != "":   # ループ終了後に, subject, relation, object, label がすべて空でない場合
        relations.append({"head":subject.strip(), "type":relation.strip(), "tail":object.strip(), "label":label})   # subject, relation, object, label を relations に追加
    return relations    # relations を返却

# テキストからリレーションのリストを抽出し, ナレッジベースに追加するメソッド
def from_text_to_kb(kb, text, label):   # verbose はログ出力の指定
    model_inputs = tokenizer(text, max_length = 256, padding = True, truncation = True, return_tensors = 'pt')  # テキストをトークン化
    gen_kwargs = {"max_length":256, "num_beams":1, "num_return_sequences":1}    # テキスト生成のパラメータを設定
    generated_tokens = model.generate(model_inputs["input_ids"].to(model.device), attention_mask = model_inputs["attention_mask"].to(model.device), **gen_kwargs)   # 設定したパラメータを用いて, モデルから生成されたトークンを取得
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens = False)   # 生成されたトークンをデコードして, 生成テキスト (特殊トークン (<triplet>, <subj>, <relation>, <obj>) と対応するトークンを含む形) のリストを作成
    for sentence in decoded_preds:  # 生成テキストを順に代入
        relations = extract_relations_from_model_output(sentence, label)    # テキストからリレーションのリストを抽出
        kb.add_relation(relations[0])   # kb オブジェクトの relations リストに追加

# クエリを実行するメソッド
def run_query(query, parameters):
    with driver.session() as session:   # セッションの開始
        result = session.run(query, parameters) # クエリの実行結果を格納
        return result.data()    # クエリの実行結果を返却

# リレーションの類似度を計算するメソッド
def compare_relations_semantically(r1, r2_list):
    sentence1 = f"{r1[0]} {r1[1]} {r1[2]}".lower().replace("_", " ")    # リレーションを文 (小文字) に変換し, _ を空白に変換
    embedding1 = model.encode(sentence1)    # 文のベクトルを取得
    max_similarity_score = 0    # 類似度 (最大値) を初期化
    for r2 in r2_list:  # リレーションを順に代入
        sentence2 = f"{r2[0]} {r2[1]} {r2[2]}".lower().replace("_", " ")    # リレーションを文 (小文字) に変換し, _ を空白に変換
        embedding2 = model.encode(sentence2)    # 文のベクトルを取得
        similarity_score = util.cos_sim(embedding1, embedding2) # 類似度を計算
        if similarity_score.item() > max_similarity_score:  # 類似度 (最大値) を更新可能な場合
            max_similarity_score = similarity_score.item()  # 類似度 (最大値) を更新
            r2_max = (r2[0], r2[1], r2[2])  # リレーションを保存
    print(f"リレーション:{r1} {r2_max}   ラベル:{r1[3]}   類似度:{max_similarity_score}")   # リレーション, ラベル, 類似度 (最大値) を表示
    return max_similarity_score # 類似度 (最大値) を返却

claim_kb = KB() # claim_kb オブジェクトを作成
supports_kb = KB()  # supports_kb オブジェクトを作成
refutes_kb = KB()   # refutes_kb オブジェクトを作成

model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large") # モデルのロード
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large") # トークナイザーのロード
with open("filtered_paper_test.jsonl", "r", encoding = "utf-8") as f:   # ファイルを読み込み専用でオープン
    for line in f:  # ファイルの各行を順に読み込み
        data = json.loads(line.strip())  # 各行を読み込み
        from_text_to_kb(claim_kb, data["claim"], data["label"]) # テキストからリレーションのリストを抽出し, ナレッジベースに追加
claim_kb.print()    # claim_kb オブジェクトの relations リストに含まれるすべてのリレーションを出力

# relations = []  # リレーションを格納するリストを初期化
# with open("relations.jsonl", "r", encoding = "utf-8") as f: # ファイルを読み込み専用でオープン
#     for line in f:  # ファイルの各行を順に読み込み
#         data = json.loads(line.strip())  # 各行を読み込み
#         relations.append({"head":data["head"], "type":data["type"], "tail":data["tail"], "label":data["label"]})    # head, type, tail, label を relations に追加
# for relation in relations:  # リレーションを順に代入
#     claim_kb.add_relation(relation) # claim_kb オブジェクトの relations リストに追加

uri = "bolt://localhost:7687" # URI
username = "username"  # あなたのユーザ名
password = "password" # あなたのパスワード
driver = GraphDatabase.driver(uri, auth = (username, password)) # セッション開始
values = [] # クエリ結果を格納するリストの初期化
model = SentenceTransformer("paraphrase-mpnet-base-v2") # SBERT モデルのロード
for c in claim_kb.relations:  # リレーションを順に代入
    subject = None  # subject を初期化
    relation = None # relation を初期化
    object = None   # object を初期化
    label = None    # label を初期化
    if "head" in c: # リレーションに "head" が存在する場合
        subject = c["head"] # subject に代入
    if "type" in c: # リレーションに "type" が存在する場合
        relation = c["type"]    # relation に代入
    if "tail" in c: # リレーションに "tail" が存在する場合
        object = c["tail"]  # object に代入
    if "label" in c:    # リレーションに "label" が存在する場合
        label = c["label"]  # label に代入
    if subject and relation and object and label:   # リレーションとラベルが空でない場合
        query = "MATCH (s)-[r]-(o) WHERE s.name = $subject RETURN DISTINCT s.name AS head, type(r) AS type, o.name AS tail" # クエリ
        parameters = {"subject":subject}    # パラメータを辞書形式で作成
        query_result = run_query(query, parameters) # クエリを実行
        if query_result:    # query_result が空でない場合
            for q in query_result:  # クエリ結果を順に代入
                key = (subject, relation, object, label)    # key(tuple) を定義
                value = (q["head"], q["type"], q["tail"])    # value(tuple) を定義
                values.append(value)    # value(tuple) を追加
            similarity_score = compare_relations_semantically(key, values)  # 類似度 (最大値) を取得
            if similarity_score >= 0.7: # 類似度 (最大値) が 0.7 以上の場合
                supports_kb.add_relation({"head":subject, "type":relation, "tail":object, "label":label})   # supports_kb オブジェクトの relations リストに追加
            else:   # 類似度 (最大値) が 0.7 未満の場合
                refutes_kb.add_relation({"head":subject, "type":relation, "tail":object, "label":label})    # refutes_kb オブジェクトの relations リストに追加
        else:   # query_result が空の場合
            print(f"not query ({subject}, {relation}, {object}, {label})")  # リレーション, ラベルを表示
            refutes_kb.add_relation({"head":subject, "type":relation, "tail":object, "label":label})    # refutes_kb オブジェクトの relations リストに追加
        values.clear()  # クエリ結果を格納するリストのクリア
driver.close()  # セッション終了

tp = 0  # (label == True(SUPPORTS)) and (prediction == Positive(supports_kb)) = 1 (True Positive:prediction == Positive が正解)
fp = 0  # (label == False(REFUTES)) and (prediction == Positive(supports_kb)) = 0 (False Positive:prediction == Positive が不正解)
tn = 0  # (label == False(REFUTES)) and (prediction == Negative(refutes_kb)) = 1 (True Negative:prediction == Negative が正解)
fn = 0  # (label == True(SUPPORTS)) and (prediction == Negative(refutes_kb)) = 0 (False Negative:prediction == Negative が不正解)
for s in supports_kb.relations: # supports_kb オブジェクトの relations リストに含まれるリレーションを順に代入
    if s["label"] == "SUPPORTS":    # label == SUPPORTS の場合
        tp += 1 # インクリメント
    else:   # label == REFUTES の場合
        fp += 1 # インクリメント
for r in refutes_kb.relations:  # refutes_kb オブジェクトの relations リストに含まれるリレーションを順に代入
    if r["label"] == "REFUTES": # label == REFUTES の場合
        tn += 1 # インクリメント
    else:   # label == SUPPORTS の場合
        fn += 1   # インクリメント

print(f"TP:{tp}")   # TP を表示
print(f"FP:{fp}")   # FP を表示
print(f"TN:{tn}")   # TN を表示
print(f"FN:{fn}")   # FN を表示
accuracy = (tp + tn) / (tp + fp + tn + fn)  # 正解率 (Accuracy) を計算
precision = tp / (tp + fp)  # 適合率 (Precision) を計算
recall = tp / (tp + fn) # 再現率 (Recall) を計算
f_measure = 2 * recall * precision / (recall + precision)   # F 値 (F-measure) を計算
print(f"正解率 (Accuracy):{accuracy}")  # 正解率 (Accuracy) を表示
print(f"適合率 (Precision):{precision}")    # 適合率 (Precision) を表示
print(f"再現率 (Recall):{recall}")  # 再現率 (Recall) を表示
print(f"F 値 (F-measure):{f_measure}")  # F 値 (F-measure) を表示
