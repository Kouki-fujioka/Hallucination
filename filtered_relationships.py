import csv

id_name = {}    # id, name を格納する辞書を初期化
with open("nodes.csv", "r", encoding = "utf-8") as n:   # ファイルを読み込み専用でオープン
    nodes_reader = csv.reader(n, delimiter = ",")   # CSV リーダー
    next(nodes_reader)  # 1 行目 (ヘッダ) をスキップ
    for row in nodes_reader:    # ファイルの各行を順に読み込み
        id, name, label = row   # id, name, label を代入
        id_name[id] = name  # key(id), value(name) の辞書を作成

with open("relationships.csv", "r", encoding = "utf-8") as r:   # ファイルを読み込み専用でオープン
    with open("filtered_relationships.csv", "w", encoding = "utf-8", newline = "") as f:    # ファイルを書き込み専用でオープン
        relationships_reader = csv.reader(r)  # CSV リーダー
        writer = csv.writer(f)  # CSV ライター
        header = next(relationships_reader) # 1 行目 (ヘッダ) を取得後, 2 行目にスキップ
        writer.writerow(header) # ヘッダ書き込み
        for row in relationships_reader:    # ファイルの各行を順に読み込み
            subject_id, relation_id, object_id = row    # id を代入
            if (subject_id in id_name) and (relation_id in id_name) and (object_id in id_name): # id が id_name に存在する場合
                relation_type = id_name[relation_id]    # relation_id の name を取得
                writer.writerow([subject_id, relation_type, object_id]) # ファイルに書き込み
