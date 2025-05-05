import bz2
import json
import codecs
import csv

# https://dumps.wikimedia.org/wikidatawiki/entities/

"""
with bz2.BZ2File('latest-all.json.bz2', 'r') as f, \
     codecs.open('nodes.csv', 'w', 'utf-8') as nc, \
     codecs.open('relationships.csv', 'w', 'utf-8') as rc:
"""
with bz2.BZ2File('latest-all.json.bz2', 'r') as f:  # 圧縮ファイルを読み込み専用でオープン
    with codecs.open('nodes.csv', 'w', 'utf-8') as n:  # ファイルを書き込み専用でオープン
        with codecs.open('relationships.csv', 'w', 'utf-8') as r:  # ファイルを書き込み専用でオープン
            ncw = csv.writer(n)    # CSV ライター
            rcw = csv.writer(r)    # CSV ライター
            ncw.writerow(['id:ID', 'name', ':LABEL'])   # ヘッダ書き込み
            rcw.writerow([':START_ID', ':TYPE', ':END_ID']) # ヘッダ書き込み
            nodes = []  # id(qid or pid), id_name, id_label(item or property) を格納するリストを初期化
            relationships = []  # リレーション (subject_id, relation_id, object_id) を格納するリストを初期化
            next(f) # 1 行目はスキップ
            for i, line in enumerate(f, 1): # ファイルの各行を順に読み込み (i = 行番号, line = 行)
                try:    # デコードエラーが発生しない場合
                    line = json.loads(line[:-2])    # 各行の末尾 2 文字を除去
                except json.decoder.JSONDecodeError:    # デコードエラーが発生した場合
                    print(f"Error at line {i}. Skipping.")  # エラーの行番号を表示
                    continue    # 50 行目までの処理をスキップ
                id = line.get('id') # id(qid or pid) を取得 (id は 1 個 / 行)
                id_name = line.get('labels').get('en').get('value') # id_name を取得
                id_label = line.get('type') # id_label(item or property) を取得
                if id and id_name and id_label: # id(qid or pid), id_name, id_label(item or property) が存在する場合
                    nodes.append((id, id_name, id_label))   # id(qid or pid), id_name, id_label(item or property) を格納するリストに追加
                    triples = [] # id に関連するトリプルを複数格納するリストを初期化
                    for _, claims in line.get('claims', {}).items(): # クレーム (id に関連する情報) を複数取得
                        for claim in claims:    # クレーム (id に関連する情報) を順に代入
                            try:    # id に関連するトリプルを取得できる場合
                                subject_id = id  # id(qid or pid) を代入
                                relation_id = claim['mainsnak']['property'] # id(pid) を取得
                                object_id = claim['mainsnak']['datavalue']['value']['id']    # id(qid or pid) を取得
                            except Exception:   # id に関連するトリプルを取得できない場合
                                continue    # 42 行目の処理をスキップ
                            triples.append((subject_id, relation_id, object_id))    # id に関連するトリプルを複数格納するリストに追加
                    relationships.extend(triples)   # リレーション (subject_id, relation_id, object_id) を格納するリストに追加
                    triples.clear() # id に関連するトリプルを複数格納するリストをクリア
                if i % 10000000 == 0:   # 行番号が 10000000 の倍数の場合
                    print(i)    # 行番号を表示
                    ncw.writerows(nodes)    # id(qid or pid), id_name, id_label(item or property) をファイルに書き込み
                    rcw.writerows(relationships)    # リレーション (subject_id, relation_id, object_id) をファイルに書き込み
                    nodes.clear()   # id(qid or pid), id_name, id_label(item or property) を格納するリストをクリア
                    relationships.clear()   # リレーション (subject_id, relation_id, object_id) を格納するリストをクリア
            ncw.writerows(nodes)    # id(qid or pid), id_name, id_label(item or property) をファイルに書き込み
            rcw.writerows(relationships)    # リレーション (subject_id, relation_id, object_id) をファイルに書き込み
            nodes.clear()   # id(qid or pid), id_name, id_label(item or property) を格納するリストをクリア
            relationships.clear()   # リレーション (subject_id, relation_id, object_id) を格納するリストをクリア
