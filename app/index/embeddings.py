# Create embeddings for law by OpenAI and Langchain in JSON format

import os
import time
import logging
import chromadb
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from multiprocessing import Manager, Pool
from tqdm import tqdm

from util.openai import embedder, text_splitter, calculate_embedding_cost
from util.tqdm import chunker, chunk_by_batch_size


# Get logger
logger = logging.getLogger(__name__)


# Define percentage of documents to be processed, 100% means all documents.
# With lower percentage, the processing time will be shorter.
# This would be useful for debugging or experimenting.
# Set to 100% for production.
PERCENTAGE_OF_DOCUMENTS_TO_BE_PROCESSED = int(os.environ.get('PERCENTAGE_OF_DOCUMENTS_TO_BE_PROCESSED', 1))


# the metadata extraction function
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["law_level"] = record.get("LawLevel")
    metadata["law_name"] = record.get("LawName")
    metadata["law_category"] = record.get("LawCategory")
    metadata["law_article_chapter"] = record.get("LawArticleChapter")
    metadata["law_article_no"] = record.get("LawArticleNo")
    metadata["law_article_content"] = record.get("LawArticleContent")

    # replace source with law url
    if record.get("LawURL"):
        metadata["source"] = record.get("LawURL")

    return metadata


# precreate the vectorstore with collection names
def precreate_vectorstore(
        vectorstore_filepath: str,
        collection_name: str):
    # create vectorstore
    langchain_chroma = Chroma(
        collection_name=collection_name,
        persist_directory=vectorstore_filepath,
        )
    # persist vectorstore
    langchain_chroma.persist()
    langchain_chroma = None


# run documents through the embeddings and add to the vectorstore.
def add_documents(
        vectorstore_filepath: str,
        collection_name: str,
        documents) -> bool:
    return_value = False

    # catch exception to prevent from crashing in multiprocessing
    try:
        # create vectorstore
        langchain_chroma = Chroma(
            collection_name=collection_name,
            embedding_function=embedder(),
            persist_directory=vectorstore_filepath,
            )
        # add documents to vectorstore
        langchain_chroma.add_documents(
            documents=documents,
            )
        # persist vectorstore
        langchain_chroma.persist()
        langchain_chroma = None
        return_value = True
    except Exception as e:
        logger.error(f'Add {documents} to {collection_name} with error: {e}')

    # debug purpose to list collection names
    vdb = chromadb.PersistentClient(path=vectorstore_filepath)
    collection_names = vdb.list_collections()
    logger.debug(f'Collection names: {collection_names}')

    return return_value


# transform embeddings for law by OpenAI and Langchain in JSON format
def transformer(
        src_filepath: str,
        vectorstore_filepath: str,
        collection_name: str = 'law',
        chunk_size=800,
        chunk_overlap=10):
    # load data
    logger.info(f'Loading JSON data from file {src_filepath}')
    loader = JSONLoader(
        file_path=src_filepath,
        jq_schema='.data[]',
        content_key='LawArticleCombinedContent',
        metadata_func=metadata_func
    )
    articles = loader.load()
    logger.info(f'Loaded {len(articles)} articles from file {src_filepath}')

    # cut articles to {PERCENTAGE_OF_DOCUMENTS_TO_BE_PROCESSED}%
    if PERCENTAGE_OF_DOCUMENTS_TO_BE_PROCESSED < 100:
        logger.info(f'Cutting articles to {PERCENTAGE_OF_DOCUMENTS_TO_BE_PROCESSED}%')
        articles = articles[:int(len(articles)*PERCENTAGE_OF_DOCUMENTS_TO_BE_PROCESSED/100)]

    logger.info(f'Ready to process {len(articles)} articles')

    # split articles into chunked documents
    logger.info(f'Splitting articles into chunked documents with chunk size {chunk_size} and overlap {chunk_overlap}')
    documents = text_splitter(articles, chunk_size, chunk_overlap)

    # save documents to file
    logger.info(f'Saving chunked documents to file {src_filepath}.documents')
    with open(f'{src_filepath}.documents', 'w', encoding='utf-8') as f:
        for document in documents:
            f.write(f'{document}\n')
    logger.info(f'Saved {len(documents)} chunked documents to file {src_filepath}.documents')

    # estimate token and cost
    total_tokens, total_cost = calculate_embedding_cost(articles)
    logger.info(f'Total tokens: {total_tokens}, total cost: USD${total_cost:.5f}')

    # ask for confirmation to proceed or not
    proceed = input('Do you want to proceed? (yes/no)')
    if proceed.lower() in ["yes", "y"]:
        logger.info('User confirmed to proceed')
    else:
        logger.info('User cancelled the process')
        return

    # calculate batch size based on total documents
    # batch size is the number of documents to be processed in one batch
    batches, batch_size = chunker(documents)
    logger.info(f'Batch size is {batch_size}, total batches is {len(batches)}')

    # determine number of worker by CPU count, and reserve one for main process
    worker_count = os.cpu_count() - 1
    logger.info(f'Worker count is {worker_count}')

    with Manager() as manager:
        # create progress bar and prepare to enqueue tasks
        logger.info('Enqueue tasks...(this may take a while)')
        pbar = tqdm(total=len(batches), desc="Processing articles")

        # precreate vectorstore with collection names
        precreate_vectorstore(vectorstore_filepath, collection_name)

        def update_progress(result):
            pbar.update(1)
            # pause 10 mini seconds to avoid too many requests
            time.sleep(0.01)

        with Pool(processes=worker_count) as pool:
            # enqueue tasks
            for batch in batches:
                pool.apply_async(add_documents,
                                (vectorstore_filepath, collection_name, batch),
                                callback=update_progress)
            # close the process pool
            pool.close()
            # wait for all tasks to finish
            pool.join()

            # pause 100 mini seconds to update progress bar
            time.sleep(0.1)

        pbar.close()

        logger.info(f'Batch {len(batches)} processed')
