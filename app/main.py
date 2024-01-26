import os
import json
import logging
import argparse

# Import proprietory module
import config.env
import config.logging


# Get logger
logger = logging.getLogger(__name__)

# Function to transform law data
def transform_law():
    from assets.law import loader, transformer

    # transform law data
    data = loader(os.environ.get('LAW_FILEPATH'))
    logger.info(f'Loaded {len(data)} records from file {os.environ.get("LAW_FILEPATH")}')

    collection = transformer(data)
    logger.info(f'Transformed {len(collection.data)} records')

    # write to file in JSON format
    collection.to_json_file(os.environ.get('LAW_FILEPATH_TRANSFORMED'), 'w', ensure_ascii=False, indent=4)
    # write to file in txt format
    with open(os.environ.get('LAW_FILEPATH_TRANSFORMED_TXT'), 'w', encoding='utf-8') as f:
        for article in collection.data:
            f.write(f'{article.LawCategory}\t{article.LawArticleChapter}\t{article.LawArticleNo}\t{article.LawArticleContent}\n')


# Function to transform order data
def transform_order():
    from assets.law import loader, transformer

    # transform order data
    data = loader(os.environ.get('ORDER_FILEPATH'))
    logger.info(f'Loaded {len(data)} records from file {os.environ.get("ORDER_FILEPATH")}')

    collection = transformer(data)
    logger.info(f'Transformed {len(collection.data)} records')

    # write to file in JSON format
    collection.to_json_file(os.environ.get('ORDER_FILEPATH_TRANSFORMED'), 'w', ensure_ascii=False, indent=4)
    # write to file in txt format
    with open(os.environ.get('ORDER_FILEPATH_TRANSFORMED_TXT'), 'w', encoding='utf-8') as f:
        for article in collection.data:
            f.write(f'{article.LawCategory}\t{article.LawArticleChapter}\t{article.LawArticleNo}\t{article.LawArticleContent}\n')


# Function to transform law embeddings
def transform_law_embeddings():
    from index.embeddings import transformer

    transformer(
        os.environ.get('LAW_FILEPATH_TRANSFORMED'),
        os.environ.get('EMBEDDINGS_LAW_FILEPATH'),
        collection_name_prefix=os.environ.get('EMBEDDINGS_LAW_COLLECTION_PREFIX'),
        collection_partition_size=int(os.environ.get('EMBEDDINGS_COLLECTION_PARTITION_SIZE')),
        chunk_size=800,
        chunk_overlap=100)


# Function to transform order embeddings
def transform_order_embeddings():
    from index.embeddings import transformer

    transformer(
        os.environ.get('ORDER_FILEPATH_TRANSFORMED'),
        os.environ.get('EMBEDDINGS_ORDER_FILEPATH'),
        collection_name_prefix=os.environ.get('EMBEDDINGS_ORDER_COLLECTION_PREFIX'),
        collection_partition_size=int(os.environ.get('EMBEDDINGS_COLLECTION_PARTITION_SIZE')),
        chunk_size=800,
        chunk_overlap=100)


# Function to get relevant documents by query
def get_relevant_documents_by_query(query: str):
    from query.embeddings import load_vector_db, create_merger_retriever, get_relevant_documents
    """
    Get relevant documents by query.
    """
    if not query:
        return "Please provide a query."

    # load vector database from disk for law
    law_vdbs = load_vector_db(
        vectorstore_filepath=os.environ.get('EMBEDDINGS_LAW_FILEPATH'),
        collection_name_prefix=os.environ.get('EMBEDDINGS_LAW_COLLECTION_PREFIX'),
        collection_partition_size=int(os.environ.get('EMBEDDINGS_COLLECTION_PARTITION_SIZE')))
    # load vector database from disk for order
    order_vdbs = load_vector_db(
        vectorstore_filepath=os.environ.get('EMBEDDINGS_ORDER_FILEPATH'),
        collection_name_prefix=os.environ.get('EMBEDDINGS_ORDER_COLLECTION_PREFIX'),
        collection_partition_size=int(os.environ.get('EMBEDDINGS_COLLECTION_PARTITION_SIZE')))
    # merge vdbs into langchain_chromas
    langchain_chromas = law_vdbs + order_vdbs
    # create merger retriever
    retriever = create_merger_retriever(langchain_chromas)
    # get relevant documents
    search_results = retriever.get_relevant_documents(query)

    return search_results


# Run the cli app with arguments
if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform-law-n-order', action='store_true',
                        help='Transform law and order data for embeddings')
    parser.add_argument('--transform-law-embeddings', action='store_true',
                        help='Transform law embeddings')
    parser.add_argument('--transform-order-embeddings', action='store_true',
                        help='Transform order embeddings')
    # get relevant documents by query
    parser.add_argument('--query', type=str, help='Query string')

    args = parser.parse_args()

    if args.transform_law_n_order:
        transform_law()
        transform_order()
    if args.transform_law_embeddings:
        transform_law_embeddings()
    if args.transform_order_embeddings:
        transform_order_embeddings()
    if args.query:
        search_results = get_relevant_documents_by_query(args.query)
        print(search_results)