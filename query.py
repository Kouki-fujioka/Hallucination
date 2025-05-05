import wikipedia
import pandas as pd
import torch
import math
from sentence_transformers import util, SentenceTransformer
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
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

# 大規模のテキストからリレーションのリストを抽出し, ナレッジベースに追加するメソッド
def from_long_text_to_kb(kb, text, span_length = 256, verbose = False):  # verbose はログ出力の指定
    model_inputs = tokenizer([text], return_tensors = "pt") # テキスト全体 (リスト形式で複数のテキストを一度に処理, PyTorch テンソルで返却) をトークン化
    num_tokens = len(model_inputs["input_ids"][0])  # トークン数
    if verbose: # verbose が True の場合
        print(f"Input has {num_tokens} tokens") # トークン数を出力
    num_spans = math.ceil(num_tokens / span_length) # スパン数
    if verbose: # verbose が True の場合
        print(f"Input has {num_spans} spans")   # スパン数を出力
    overlap = math.ceil((num_spans * span_length - num_tokens) / max(num_spans - 1, 1)) # スパン間の重複部分の長さを計算 (トークン数がスパン長の倍数の場合は 0)
    spans_boundaries = []   # 各スパンの開始位置と終了位置を格納するリストを初期化
    start = 0   # 開始位置を初期化
    for i in range(num_spans):  # i = 0 ~ num_spans - 1
        spans_boundaries.append([start + span_length * i, start + span_length * (i + 1)])   # spans_boundaries リストに追加
        start -= overlap    # 開始位置を調整 (トークン数がスパン長の倍数でない場合に, 重複部分を設けて情報の欠落を防ぐ)
    if verbose: # verbose が True の場合
        print(f"Span boundaries are {spans_boundaries}")    # スパン境界を出力
    tensor_ids = [model_inputs["input_ids"][0][boundary[0]:boundary[1]] for boundary in spans_boundaries]   # 各スパンに対応するトークン id を抽出
    tensor_masks = [model_inputs["attention_mask"][0][boundary[0]:boundary[1]] for boundary in spans_boundaries]    # 各スパンに対応するアテンションマスクを抽出
    model_inputs = {"input_ids":torch.stack(tensor_ids), "attention_mask":torch.stack(tensor_masks)}    # 抽出したトークン id とアテンションマスクの辞書を作成
    gen_kwargs = {"max_length":256, "length_penalty":0, "num_beams":3, "num_return_sequences":3}    # テキスト生成のパラメータを設定
    generated_tokens = model.generate(**model_inputs, **gen_kwargs)   # 設定したパラメータを用いて, モデルから生成されたトークンを取得
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens = False)   # 生成されたトークンをデコードして, 生成テキスト (特殊トークン (<triplet>, <subj>, <relation>, <obj>) と対応するトークンを含む形) のリストを作成
    i = 0   # インデックス変数を初期化
    for sentence_pred in decoded_preds: # 生成テキストを順に代入
        relations = extract_relations_from_model_output(sentence_pred)  # 分割したテキストからリレーションのリストを抽出
        for relation in relations:  # リレーションを順に代入
            kb.add_relation(relation)   # kb オブジェクトの relations リストに追加
        i += 1  # インクリメント

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

pipe = pipeline("text-generation", model = "HuggingFaceH4/zephyr-7b-beta", torch_dtype = torch.bfloat16, device_map = "auto")   # パイプラインを作成
messages = [
    {
        "role":"system",    # 設定
        "content":"You are a question-answering assistant"  # 設定内容
    },
    {
        "role":"user",  # 入力
        "content":"Tell me about Napoleon Bonaparte's lifespan" # 入力内容
    }
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)   # プロンプトを生成
outputs = pipe(prompt, max_new_tokens = 256, temperature = 0)   # テキストを生成
text = outputs[0]["generated_text"] # テキストを取得
print(text) # テキストを表示

model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large") # モデルのロード
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large") # トークナイザーのロード
llm_kb = KB()   # llm_kb オブジェクトを作成
fact_kb = KB()  # fact_kb オブジェクトを作成
hallucination_kb = KB() # hallucination_kb オブジェクトを作成
from_long_text_to_kb(llm_kb, text, verbose = True)  # 大規模のテキストからリレーションのリストを抽出し, ナレッジベースに追加
llm_kb.print()  # llm_kb オブジェクトの relations リストに含まれるすべてのリレーションを出力

uri = "bolt://localhost:7687" # URI
username = "username"  # あなたのユーザ名
password = "password" # あなたのパスワード
driver = GraphDatabase.driver(uri, auth = (username, password)) # セッション開始
values = [] # クエリ結果を格納するリストの初期化
model = SentenceTransformer("paraphrase-mpnet-base-v2") # SBERT モデルのロード
for l in llm_kb.relations:  # リレーションを順に代入
    subject = None  # subject を初期化
    relation = None # relation を初期化
    object = None   # object を初期化
    if "head" in l: # リレーションに "head" が存在する場合
        subject = l["head"] # subject に代入
    if "type" in l: # リレーションに "type" が存在する場合
        relation = l["type"]    # relation に代入
    if "tail" in l: # リレーションに "tail" が存在する場合
        object = l["tail"]  # object に代入
    if subject and relation and object: # リレーションが空でない場合
        query = "MATCH (s)-[r]-(o) WHERE s.name = $subject RETURN DISTINCT s.name AS head, type(r) AS type, o.name AS tail" # クエリ
        parameters = {"subject":subject}    # パラメータを辞書形式で作成
        query_result = run_query(query, parameters) # クエリを実行
        if query_result:    # query_result が空でない場合
            for q in query_result:  # クエリ結果を順に代入
                key = (subject, relation, object)   # key(tuple) を定義
                value = (q["head"], q["type"], q["tail"])    # value(tuple) を定義
                values.append(value)    # value(tuple) を追加
            similarity_score = compare_relations_semantically(key, values)  # 類似度 (最大値) を取得
            if similarity_score >= 0.7: # 類似度 (最大値) が 0.7 以上の場合
                fact_kb.add_relation({"head":subject, "type":relation, "tail":object})  # fact_kb オブジェクトの relations リストに追加
            else:   # 類似度 (最大値) が 0.7 未満の場合
                hallucination_kb.add_relation({"head":subject, "type":relation, "tail":object}) # hallucination_kb オブジェクトの relations リストに追加
        else:   # query_result が空の場合
            print(f"not query ({subject}, {relation}, {object})")   # リレーションを表示
            hallucination_kb.add_relation({"head":subject, "type":relation, "tail":object}) # hallucination_kb オブジェクトの relations リストに追加
        values.clear()  # クエリ結果を格納するリストのクリア
driver.close()  # セッション終了

fact_kb.print() # fact_kb オブジェクトの relations リストに含まれるすべてのリレーションを出力
hallucination_kb.print()    # hallucination_kb オブジェクトの relations リストに含まれるすべてのリレーションを出力
