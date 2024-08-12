import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict, Any
import os
from datetime import datetime

class SearchXDatasetBuilder:
    def __init__(self, api_url: str, output_dir: str = 'search_datasets'):
        self.api_url = api_url
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def search(self, query: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform a search query using SearchX API.
        
        :param query: The search query string
        :param options: A dictionary of search options
        :return: A list of search result dictionaries
        """
        params = {
            'q': query,
            **options
        }
        response = requests.get(self.api_url, params=params)
        response.raise_for_status()
        return response.json()['results']

    def build_dataset(self, queries: List[str], options_list: List[Dict[str, Any]]) -> str:
        """
        Build a dataset by performing multiple searches and storing results in a Parquet file.
        
        :param queries: A list of search queries
        :param options_list: A list of dictionaries, each containing search options for a query
        :return: Path to the created Parquet file
        """
        all_results = []

        for query, options in zip(queries, options_list):
            results = self.search(query, options)
            for result in results:
                result['query'] = query
                result['search_options'] = str(options)
            all_results.extend(results)

        df = pd.DataFrame(all_results)
        
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_results_{timestamp}.parquet"
        filepath = os.path.join(self.output_dir, filename)

        # Write to Parquet file
        table = pa.Table.from_pandas(df)
        pq.write_table(table, filepath)

        return filepath

# Usage example
if __name__ == "__main__":
    builder = SearchXDatasetBuilder(api_url="https://searchx-api-example.com/search")
    
    queries = [
        "machine learning",
        "artificial intelligence",
        "data science"
    ]
    
    options_list = [
        {"num_results": 10, "search_engine": "bing"},
        {"num_results": 15, "search_engine": "google"},
        {"num_results": 20, "search_engine": "duckduckgo"}
    ]
    
    dataset_path = builder.build_dataset(queries, options_list)
    print(f"Dataset created: {dataset_path}")