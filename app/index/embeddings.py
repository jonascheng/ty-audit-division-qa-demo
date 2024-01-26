# Create embeddings for law by OpenAI and Langchain in JSON format

import os
import time
import logging
import chromadb
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from multiprocessing import Manager, Pool
from tqdm import tqdm

from util.openai import embedder
from util.tqdm import chunker, chunk_by_batch_size


# Get logger
logger = logging.getLogger(__name__)


# Define percentage of documents to be processed, 100% means all documents.
# With lower percentage, the processing time will be shorter.
# This would be useful for debugging or experimenting.
# Set to 100% for production.
PERCENTAGE_OF_DOCUMENTS_TO_BE_PROCESSED = 1

# the metadata extraction function
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["law_level"] = record.get("LawLevel")
    metadata["law_name"] = record.get("LawName")
    metadata["law_category"] = record.get("LawCategory")
    metadata["law_article_chapter"] = record.get("LawArticleChapter")
    metadata["law_article_no"] = record.get("LawArticleNo")

    # replace source with law url
    if record.get("LawURL"):
        metadata["source"] = record.get("LawURL")

    return metadata


# split text into documents
def splitter(articles, chunk_size=900, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(
        # separators=["\n\n", "\r\n", "\r", "\n", " ", "。", "　"],
        separators=[r"\s+", "。", "　", "＞"],
        is_separator_regex=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(articles)
    return documents


# run documents through the embeddings and add to the vectorstore.
def add_documents(
        vectorstore_filepath: str,
        queue,
        documents) -> bool:
    # get collection name from queue
    collection_name = queue.get()
    logger.debug(f'Use collection name {collection_name} from queue')

    # create vectorstore
    langchain_chroma = Chroma(
        collection_name=collection_name,
        embedding_function=embedder(),
        persist_directory=vectorstore_filepath,
        )
    langchain_chroma.add_documents(
        documents=documents,
        )
    langchain_chroma.persist()
    langchain_chroma = None

    # put collection name back to queue
    queue.put(collection_name)
    logger.debug(f'Collection name {collection_name} returned to queue')

    # debug purpose to list collection names
    vdb = chromadb.PersistentClient(path=vectorstore_filepath)
    collection_names = vdb.list_collections()
    logger.debug(f'Collection names: {collection_names}')

    return True


# transform embeddings for law by OpenAI and Langchain in JSON format
def transformer(
        src_filepath: str,
        vectorstore_filepath: str,
        collection_name_prefix: str = 'law',
        collection_partition_size: int = 4,
        chunk_size=900,
        chunk_overlap=100):
    # load data
    logger.info(f'Loading data from file {src_filepath}')
    loader = JSONLoader(
        file_path=src_filepath,
        jq_schema='.data[]',
        content_key='LawArticleContent',
        metadata_func=metadata_func
    )
    articles = loader.load()
    logger.info(f'Loaded {len(articles)} records from file {src_filepath}')

    # cut articles to {PERCENTAGE_OF_DOCUMENTS_TO_BE_PROCESSED}%
    if PERCENTAGE_OF_DOCUMENTS_TO_BE_PROCESSED < 100:
        logger.info(f'Cutting articles to {PERCENTAGE_OF_DOCUMENTS_TO_BE_PROCESSED}%')
        articles = articles[:int(len(articles)*PERCENTAGE_OF_DOCUMENTS_TO_BE_PROCESSED/100)]

    logger.info(f'Ready to process {len(articles)} records')

    # split articles into documents
    documents = splitter(articles, chunk_size, chunk_overlap)

    # save documents to file
    logger.info(f'Saving documents to file {src_filepath}.documents')
    with open(f'{src_filepath}.documents', 'w', encoding='utf-8') as f:
        for document in documents:
            f.write(f'{document}\n')

    # calculate batch size based on total documents
    # batch size is the number of documents to be processed in one batch
    batches, batch_size = chunker(documents)
    logger.info(f'Batch size is {batch_size}, total batches is {len(batches)}')

    with Manager() as manager:
        # create the shared queue for each collection partition
        queue = manager.Queue()

        for i in range(collection_partition_size):
            # create collection name
            collection_name = f'{collection_name_prefix}_{i}'
            # put collection name into queue
            logger.info(f'Put collection name {collection_name} into queue')
            queue.put(collection_name)

        # output queue status and size
        qsize = queue.qsize()
        logger.debug(f'Queue size is {qsize}')

        # create progress bar and prepare to enqueue tasks
        logger.info(f'Enqueue tasks...(this may take a while)')
        pbar = tqdm(total=len(batches), desc="Processing articles")

        def update_progress(result):
            pbar.update(1)
            # pause 100 mini seconds to avoid too many requests
            time.sleep(0.1)

        with Pool(processes=collection_partition_size) as pool:
            # enqueue tasks
            for batch in batches:
                pool.apply_async(add_documents,
                                (vectorstore_filepath, queue, batch),
                                callback=update_progress)
            # close the process pool
            pool.close()
            # wait for all tasks to finish
            pool.join()

            # pause 100 mini seconds to update progress bar
            time.sleep(0.1)

        pbar.close()

        logger.info(f'Batch {len(batches)} processed')
