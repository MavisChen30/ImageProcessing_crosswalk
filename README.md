檔案執行順序：
1. cat dataset* > datasets
2. 更改crosswalk.json中絕對路徑
3. python extract_features -c crosswalk.json
4. python train_model.py -c crosswalk.json
5. python test_model_vedio.py -c crosswalk.json
