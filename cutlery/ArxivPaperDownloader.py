import arxiv
import requests
import os
import tarfile
import gzip
import shutil

class ArxivPaperDownloader:
    def __init__(self, download_dir='papers'):
        self.download_dir = download_dir
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def search_and_download(self, arxiv_url):
        try:
            # Extract the arXiv ID from the URL
            arxiv_id = self._extract_arxiv_id(arxiv_url)
            
            # Search for the paper using the arXiv ID
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())
            
            # Download the LaTeX source
            return self._download_latex_source(paper)
        except Exception as e:
            return f"Error: {str(e)}"

    def _extract_arxiv_id(self, url):
        # Extract the arXiv ID from the URL
        parts = url.split('/')
        return parts[-1]

    def _download_latex_source(self, paper):
        try:
            # Construct the source URL
            source_url = f"https://arxiv.org/e-print/{paper.get_short_id()}"
            
            # Create a filename for the source
            filename = f"{paper.get_short_id().split('v')[0]}_source"
            filepath = os.path.join(self.download_dir, filename)
            
            # Download the source file
            response = requests.get(source_url)
            response.raise_for_status()
            
            # Save the downloaded file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Extract the contents
            if tarfile.is_tarfile(filepath):
                with tarfile.open(filepath, 'r') as tar:
                    tar.extractall(path=filepath + '_extracted')
                os.remove(filepath)
                return f"LaTeX source extracted to: {filepath}_extracted"
            elif filepath.endswith('.gz'):
                with gzip.open(filepath, 'rb') as f_in:
                    with open(filepath[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(filepath)
                return f"LaTeX source extracted to: {filepath[:-3]}"
            else:
                return f"Source file downloaded to: {filepath}"
        except Exception as e:
            return f"Error downloading LaTeX source: {str(e)}"

# Usage example
if __name__ == "__main__":
    downloader = ArxivPaperDownloader()
    result = downloader.search_and_download("https://arxiv.org/abs/2303.08774")
    print(result)
    
    
# from ArxivPaperDownloader import ArxivPaperDownloader

# downloader = ArxivPaperDownloader(download_dir='your_custom_directory')
# result = downloader.search_and_download("https://arxiv.org/abs/2303.08774")
# print(result)