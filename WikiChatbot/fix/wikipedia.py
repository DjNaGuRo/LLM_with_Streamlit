from langchain.schema import Document
import wikipedia 
from typing import List

# Overwrite the lazy_load method
def lazy_load(self, query: str) -> List[Document]:
        """
        Load documents lazily.
        """
        page_titles = self.wiki_client.search(
            query, results=self.top_k_results
        ) # Removed slice from query and added results parameter

        for page_title in page_titles:
            try:
                # Note: Calling wikipedia.page here to fetch content and build the Document object
                page = wikipedia.page(page_title, auto_suggest=False) # Use wikipedia.page from the wikipedia library and disable auto_suggest
                if page:
                    yield Document(
                        page_content=page.content,
                        metadata={"source": page.url, "title": page.title},
                    )
                else:
                    continue
            except (
                wikipedia.exceptions.PageError,
                wikipedia.exceptions.DisambiguationError,
            ):
                continue
