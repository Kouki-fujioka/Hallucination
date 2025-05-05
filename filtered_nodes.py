import csv

relationships_id = set()    #  セット (重複排除リスト) を初期化
with open("relationships.csv", mode = "r", encoding = "utf-8") as r:    # ファイルを読み込み専用でオープン
    relationships_reader = csv.reader(r, delimiter = ",")   # CSV リーダー
    next(relationships_reader)  # 1 行目 (ヘッダ) をスキップ
    for row in relationships_reader:    # ファイルの各行を順に読み込み
        subject_id, relation_id, object_id = row    # id を代入
        relationships_id.update([subject_id, relation_id, object_id])   # id をセットに追加

with open("nodes.csv", mode = "r", encoding = "utf-8") as n:    # ファイルを読み込み専用でオープン
    with open("filtered_nodes.csv", mode = "w", encoding = "utf-8", newline = "") as f: # ファイルを書き込み専用でオープン
        nodes_reader = csv.reader(n, delimiter = ",")   # CSV リーダー
        writer = csv.writer(f, delimiter = ",") # CSV ライター
        header = next(nodes_reader) # 1 行目 (ヘッダ) を取得後, 2 行目にスキップ
        writer.writerow(header)  # ヘッダ書き込み
        for row in nodes_reader:    # ファイルの各行 (id, name) を順に読み込み
            nodes_id = row[0]   # 各行の 1 列目 (id) を取得
            if nodes_id in relationships_id:    # id(nodes.csv) が relationships.csv に存在する場合 
                writer.writerow(row)    # ファイルの各行 (id, name) を書き込み
