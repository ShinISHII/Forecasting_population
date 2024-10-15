import os
import re

class MetricsExtractor:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.mape, self.mse, self.rmse = self.search_md_files_and_extract_values()

    def search_md_files_and_extract_values(self):
        mape_list = []
        mse_list = []
        rmse_list = []

        # .mdファイル内の任意の値を抽出する正規表現パターン
        mape_pattern = re.compile(r'MAPE\s+(\d+\.\d+)')
        mse_pattern = re.compile(r'MSE\s+(\d+\.\d+)')

        # 再帰的にフォルダを探索
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.md'):
                    md_file_path = os.path.join(root, file)
                    with open(md_file_path, 'r') as md_file:
                        content = md_file.read()

                        # 任意の値を正規表現で抽出
                        mape_match = mape_pattern.search(content)
                        mse_match = mse_pattern.search(content)

                        if mape_match and mse_match:
                            # 抽出した値をリストに保存
                            mape = float(mape_match.group(1))
                            mse = float(mse_match.group(1))
                            rmse = mse ** 0.5  # RMSEの計算

                            mape_list.append(mape)
                            mse_list.append(mse)
                            rmse_list.append(rmse)

        return mape_list, mse_list, rmse_list