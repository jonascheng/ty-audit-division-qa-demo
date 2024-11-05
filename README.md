# Data loader

1. 法條，透過 `JSONLoader` 載入 `*.json` 檔案
2. 調查報告，透過 `UnstructuredWordDocumentLoader` 載入 `*.doc*` 檔案
3. 新聞，透過 `UnstructuredWordDocumentLoader` 載入 `*.doc*` 檔案

Note: 未針對 `*.doc*` 檔案內註解表格等進行額外處理

# Chunking

* Normalization: Replace CJK whitespace characters, split the document into paragraphs based on the document properties defined in `separators`
* chunk_size: 800
* chunk_overlap: 100

# Embedding

* Multi-tasking
* Embedding cost estimation with consent
* Leverage model `text-embedding-3-large` for embedding
* Leverage `chromadb` to store embeddings

# Retrieval QA

* Leverage `gpt-4o` for retrieval QA
* Leverage `MultiQueryRetriever` automates the process of prompt tuning by using an LLM to generate multiple queries from different perspectives for a given user input query.
* Leverage `PromptTemplate` for prompt engineering to generate multiple queries from different perspectives for a given user input query.

```
You are an AI language model assistant. Your task is to generate 5
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions in Traditional Chinese and separated by newlines.
Original question: {question}
```

* Different `chain_type` as well as customized prompt respectively for

  * 法條: `stuff`

  * 調查報告: `refined`

  * 新聞: `refined`

* Leverage `RetrievalQA` to chain for question-answering against an index.