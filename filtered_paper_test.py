import json

# https://fever.ai/dataset/fever.html

input_file = "paper_test.jsonl" # 入力ファイル
output_file = "filtered_paper_test.jsonl"   # 出力ファイル

supports_data = []  # リストを初期化
refutes_data = []   # リストを初期化

with open(input_file, "r", encoding = "utf-8") as i:    # ファイルを読み込み専用でオープン
    for line in i:  # ファイルの各行を順に読み込み
        data = json.loads(line) # 各行を読み込み
        if data.get("label") != "NOT ENOUGH INFO":  # label != NOT ENOUGH INFO の場合
            if data["label"] == "SUPPORTS" and len(supports_data) < 50: # (label == SUPPORTS) and (要素数 < 50) の場合
                supports_data.append(data)  # リストに追加
            elif data["label"] == "REFUTES" and len(refutes_data) < 50: # (label == REFUTES) and (要素数 < 50) の場合
                refutes_data.append(data)   # リストに追加

with open(output_file, "w", encoding = "utf-8") as o:   # ファイルを書き込み専用でオープン
    for line in supports_data + refutes_data:   # リストの各行を順に読み込み
        o.write(json.dumps(line, ensure_ascii = False) + "\n")  # ファイルに書き込み
