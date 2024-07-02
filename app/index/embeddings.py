# Create embeddings for law by OpenAI and Langchain in JSON format

import os
import abc
import time
import logging
import chromadb
from langchain_community.vectorstores import Chroma
from multiprocessing import Pool
from tqdm import tqdm

from util.openai import embedder, text_splitter, calculate_embedding_cost
from util.tqdm import chunker


# Get logger
logger = logging.getLogger(__name__)


# A base class to create embeddings
class Embeddings(metaclass=abc.ABCMeta):
    def __init__(
            self,
            src_filepath: str,
            vectorstore_filepath: str,
            collection_name: str = 'law',
            chunk_size: int = 800,
            chunk_overlap: int = 10,):
        self.src_filepath = src_filepath
        self.vectorstore_filepath = vectorstore_filepath
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Define percentage of documents to be processed, 100% means all documents.
        # With lower percentage, the processing time will be shorter.
        # This would be useful for debugging or experimenting.
        # Set to 100% for production.
        self.percentage_of_documents_to_be_processed = int(
            os.environ.get('PERCENTAGE_OF_DOCUMENTS_TO_BE_PROCESSED', 1))

    # Abstract function to load documents from source filepath
    @abc.abstractmethod
    def _loader(self) -> list:
        return NotImplementedError

    # Abstract function to split documents into chunked documents
    @abc.abstractmethod
    def _splitter(self, documents: list) -> list:
        return NotImplementedError

    # replace CJK space with normal space
    def _place_cjk_space(
            self,
            documents: list) -> list:
        # iterate documents and
        # replace CJK space with normal space in page_content
        for doc in documents:
            doc.page_content = doc.page_content.replace(
                '　', ' ').replace('\u3000', ' ')
        return documents

    # initial vectorstore with collection names
    def _init_vectorstore(self):
        # create vectorstore
        store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.vectorstore_filepath,
        )
        # persist vectorstore
        store.persist()
        store = None

    # add documents to vectorstore
    def _add_documents(
            self,
            documents: list) -> bool:
        return_value = False

        # catch exception to prevent from crashing in multiprocessing
        try:
            # create vectorstore
            store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embedder(),
                persist_directory=self.vectorstore_filepath,
            )
            # add documents to vectorstore
            store.add_documents(
                documents=documents,
            )
            # persist vectorstore
            store.persist()
            store = None
            return_value = True
        except Exception as e:
            logger.error(
                f'Add {documents} to {self.collection_name} with error: {e}')

        # debug purpose to list collection names
        vdb = chromadb.PersistentClient(path=self.vectorstore_filepath)
        collection_names = vdb.list_collections()
        logger.debug(f'Collection names: {collection_names}')

        return return_value

    # entry point to run the process
    def run(self):
        # load documents
        logger.info(f'Loading data from {self.src_filepath}')
        documents = self._loader()
        logger.info(
            f'Loaded {len(documents)} documents from {self.src_filepath}')

        # check length and output the first document for debugging purpose
        if len(documents) > 0:
            logger.debug(f'First document: {documents[0]}')
        else:
            logger.info('No document loaded, exit')
            return

        # cut documents to {self.percentage_of_documents_to_be_processed}%
        if self.percentage_of_documents_to_be_processed < 100:
            logger.info(
                f'Cutting documents to {self.percentage_of_documents_to_be_processed}%')
            documents = documents[:int(
                len(documents)*self.percentage_of_documents_to_be_processed/100)]

        # replace CJK space with normal space
        logger.info('Replacing CJK space with normal space')
        documents = self._place_cjk_space(documents)

        logger.info(f'Ready to process {len(documents)} documents')

        # split documents into chunked documents
        logger.info(
            f'Splitting documents into chunked documents with chunk size {self.chunk_size} and overlap {self.chunk_overlap}')
        chunked_documents = self._splitter(documents)
        # output the first chunked document for debugging purpose
        logger.debug(f'First chunked document: {chunked_documents[0]}')

        # save chunked documents to file
        logger.info(
            f'Saving chunked documents to file {self.src_filepath}.documents')
        with open(f'{self.src_filepath}.documents', 'w', encoding='utf-8') as f:
            for document in chunked_documents:
                f.write(f'{document}\n')

        # estimate token and cost
        total_tokens, total_cost = calculate_embedding_cost(documents)
        logger.info(
            f'Total tokens: {total_tokens}, total cost: USD${total_cost:.5f}')

        # ask for confirmation to proceed or not
        proceed = input('Do you want to proceed? (yes/no)')
        if proceed.lower() in ["yes", "y"]:
            logger.info('User confirmed to proceed')
        else:
            logger.info('User cancelled the process')
            return

        # calculate batch size based on total documents
        # batch size is the number of documents to be processed in one batch
        batches, batch_size = chunker(chunked_documents)
        logger.info(
            f'Batch size is {batch_size}, total batches is {len(batches)}')

        # determine number of worker by CPU count, and reserve one for main process
        worker_count = os.cpu_count() - 1
        logger.info(f'Worker count is {worker_count}')

        # create progress bar and prepare to enqueue tasks
        logger.info('Enqueue tasks...(this may take a while)')
        pbar = tqdm(total=len(batches))

        # precreate vectorstore with collection names
        self._init_vectorstore()

        def update_progress(result):
            pbar.update(1)
            # pause 10 mini seconds to avoid too many requests
            time.sleep(0.01)

        def update_error(error):
            logger.error(f'Error: {error}')

        with Pool(processes=worker_count) as pool:
            # enqueue tasks
            for batch in batches:
                pool.apply_async(self._add_documents,
                                 (batch, ),
                                 callback=update_progress,
                                 error_callback=update_error,)
            # close the process pool
            pool.close()
            # wait for all tasks to finish
            pool.join()

            # pause 100 mini seconds to update progress bar
            time.sleep(0.1)

        pbar.close()

        logger.info(f'Batch {len(batches)} processed')


# A class to create embeddings for law in JSON format
class LawEmbeddings(Embeddings):
    # custom function to load documents from source filepath
    def _loader(self) -> list:
        from langchain_community.document_loaders import (
            DirectoryLoader,
            JSONLoader
        )

        # enumerate the source filepath, and
        # only load .json file into a list of Document objects
        loader = DirectoryLoader(
            self.src_filepath,
            glob="*.json",
            loader_cls=JSONLoader,
            loader_kwargs={
                "jq_schema": '.data[]',
                "content_key": 'LawArticleContent',
                "metadata_func": self._metadata_func},
            show_progress=True,)
        return loader.load()

    # custom function to split documents into chunked documents
    def _splitter(self, documents: list) -> list:
        documents = text_splitter(
            documents,
            self.chunk_size,
            self.chunk_overlap,
            separators=["\n\n", "\r\n", "\n", "。", "？", "："])
        # iterate documents and augment documents with metadata
        for doc in documents:
            law_name = doc.metadata["law_name"]
            law_category = doc.metadata["law_category"]
            law_article_chapter = doc.metadata["law_article_chapter"]
            law_article_no = doc.metadata["law_article_no"]
            law_article_content = doc.page_content
            # augmented article content
            augmented_article_content = f"法規名稱：{law_name}\n法規類別：{law_category}\n條文內容：{law_article_chapter}\n{law_article_no}：{law_article_content}"
            # update page content
            doc.page_content = augmented_article_content
        return documents

    # the metadata extraction function
    def _metadata_func(
            self,
            record: dict,
            metadata: dict) -> dict:
        metadata["law_level"] = record.get("LawLevel")
        metadata["law_name"] = record.get("LawName")
        metadata["law_category"] = record.get("LawCategory")
        metadata["law_article_chapter"] = record.get("LawArticleChapter")
        metadata["law_article_no"] = record.get("LawArticleNo")

        # replace source with law url
        if record.get("LawURL"):
            metadata["source"] = record.get("LawURL")

        return metadata


# A class to create embeddings for investigation reports in doc format
class InvestigationReportEmbeddings(Embeddings):
    # custom function to load documents from source filepath
    def _loader(self) -> list:
        from langchain_community.document_loaders import (
            DirectoryLoader,
            UnstructuredWordDocumentLoader
        )

        # enumerate the source filepath, and
        # only load .doc file into a list of Document objects
        loader = DirectoryLoader(
            self.src_filepath,
            glob="*.doc*",
            loader_cls=UnstructuredWordDocumentLoader,
            show_progress=True,)
        return loader.load()

    # custom function to split documents into chunked documents
    def _splitter(self, documents: list) -> list:
        return text_splitter(
            documents,
            self.chunk_size,
            self.chunk_overlap,
            separators=["\n\n", "\n", "。", "："])


# A class to create embeddings for news in doc format
class NewsEmbeddings(Embeddings):
    # override init function
    def __init__(
            self,
            src_filepath: str,
            vectorstore_filepath: str,
            collection_name: str = 'news',
            chunk_size: int = 800,
            chunk_overlap: int = 100,):
        super().__init__(
            src_filepath,
            vectorstore_filepath,
            collection_name,
            chunk_size,
            chunk_overlap,)

    # custom function to load documents from source filepath
    def _loader(self) -> list:
        from langchain_community.document_loaders import (
            DirectoryLoader,
            UnstructuredWordDocumentLoader
        )

        # enumerate the source filepath, and
        # only load .doc file into a list of Document objects
        loader = DirectoryLoader(
            self.src_filepath,
            glob="*.doc*",
            loader_cls=UnstructuredWordDocumentLoader,
            show_progress=True,)
        return loader.load()

    # custom function to split documents into chunked documents
    def _splitter(self, documents: list) -> list:
        return text_splitter(
            documents,
            self.chunk_size,
            self.chunk_overlap,
            separators=["\n\n", "\n", "。", "："])