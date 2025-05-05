1, 2 は filtered_nodes.csv, filtered_relationships.csv ファイルが存在する場合は実行しなくて良い.
6, 7 は filtered_paper_test.jsonl ファイルが存在する場合は実行しなくて良い.

1. https://dumps.wikimedia.org/wikidatawiki/entities/ latest-all.json.bz2 をダウンロード (2024/10/17)
2. filtered_nodes.py, filtered_relationships.py を実行 (filtered_nodes.csv, filtered_relationships.csv を生成)
3. filtered_nodes.csv, filtered_relationships.csv を Neo4j にインポート

提案手法の実行
4. query.py を実行

評価実験の実行 (REBEL)
5. rebel_evaluation.py を実行
生成したトリプルの数 : 261
完全一致したトリプルの数 : 171
正解率 (Accuracy) : 0.655

評価実験の実行 (提案手法(REBEL))
6. https://fever.ai/dataset/fever.html Paper Test Dataset をダウンロード
7. filtered_paper_test.py を実行
8. evaluation.py を実行
正解率 (Accuracy) : 0.590
適合率 (Precision) : 0.573
再現率 (Recall) : 0.700
F 値 (F-measure) : 0.630

評価実験の実行 (提案手法(手動))
9. evaluation.py の 121~127 行目をアンコメント
10. evaluation.py の 46~76, 78~86, 113~119 行目をコメントアウト
11. evaluation.py を実行
正解率 (Accuracy) : 0.750
適合率 (Precision) : 0.735
再現率 (Recall) : 0.780
F 値 (F-measure) : 0.757

relations.jsonl は filtered_paper_test.jsonl の claim から手動でトリプル抽出した結果を保存
rebel-corrected.txt は filtered_paper_test.jsonl の calim から REBEL でトリプル抽出した結果と変更箇所 (手動でトリプル抽出した結果を参照) を保存
