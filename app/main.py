import os
import logging
import argparse

# Import proprietory module
import config.env
import config.logging


# Get logger
logger = logging.getLogger(__name__)


# Transform law data for creating embeddings
def transform_law():
    from assets.transform import loader, transformer

    # transform law data
    data = loader(os.environ.get('LAW_FILEPATH'))
    logger.info(
        f'Loaded {len(data)} records from file {os.environ.get("LAW_FILEPATH")}')

    collection = transformer(
        data,
        allowed_category=['衛生福利部＞食品藥物管理目'])
    logger.info(f'Transformed {len(collection.data)} records')

    # get output path from env 'LAW_TRANSFORMED_PATH'
    path = os.environ.get('LAW_TRANSFORMED_PATH')
    # create full path if not exists
    os.makedirs(path, exist_ok=True)
    # join path with file name of 'LAW_FILEPATH'
    filepath = os.path.join(path, os.path.basename(
        os.environ.get('LAW_FILEPATH')))
    # write to file in JSON format
    collection.to_json_file(
        filepath, 'w',
        ensure_ascii=False,
        indent=4)


# Transform order data for creating embeddings
def transform_order():
    from assets.transform import loader, transformer

    # transform order data
    data = loader(os.environ.get('ORDER_FILEPATH'))
    logger.info(
        f'Loaded {len(data)} records from file {os.environ.get("ORDER_FILEPATH")}')

    collection = transformer(
        data,
        allowed_category=['衛生福利部＞食品藥物管理目'])
    logger.info(f'Transformed {len(collection.data)} records')

    # get output file path from env 'ORDER_TRANSFORMED_PATH'
    path = os.environ.get('ORDER_TRANSFORMED_PATH')
    # create full path if not exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # join path with file name of 'ORDER_FILEPATH'
    filepath = os.path.join(path, os.path.basename(
        os.environ.get('ORDER_FILEPATH')))
    # write to file in JSON format
    collection.to_json_file(
        filepath, 'w',
        ensure_ascii=False,
        indent=4)


# Create law embeddings
def create_law_embeddings():
    from index.embeddings import LawEmbeddings

    LawEmbeddings(
        os.environ.get('LAW_TRANSFORMED_PATH'),
        os.environ.get('EMBEDDINGS_FILEPATH'),
        collection_name=os.environ.get(
            'EMBEDDINGS_TAIWAN_LAW_COLLECTION_NAME'),
        chunk_size=800,
        chunk_overlap=10
    ).run()


# Create order embeddings
def create_order_embeddings():
    from index.embeddings import LawEmbeddings

    LawEmbeddings(
        os.environ.get('ORDER_TRANSFORMED_PATH'),
        os.environ.get('EMBEDDINGS_FILEPATH'),
        collection_name=os.environ.get(
            'EMBEDDINGS_TAIWAN_LAW_COLLECTION_NAME'),
        chunk_size=800,
        chunk_overlap=10
    ).run()


# Create investigation report embeddings
def create_investigation_embeddings():
    from index.embeddings import InvestigationReportEmbeddings

    InvestigationReportEmbeddings(
        os.environ.get('INVESTIGATION_REPORTS_PATH'),
        os.environ.get('EMBEDDINGS_FILEPATH'),
        collection_name=os.environ.get(
            'EMBEDDINGS_INVESTIGATION_REPORTS_COLLECTION_NAME'),
        chunk_size=800,
        chunk_overlap=100
    ).run()


# Return indexer base on target name
def get_indexer(target_name: str):
    from query.embeddings import QueryEmbeddings

    if target_name == 'investigation':
        collection_name = os.environ.get(
            'EMBEDDINGS_INVESTIGATION_REPORTS_COLLECTION_NAME')
    else:
        collection_name = os.environ.get(
            'EMBEDDINGS_TAIWAN_LAW_COLLECTION_NAME')

    return QueryEmbeddings(
        vectorstore_filepath=os.environ.get('EMBEDDINGS_FILEPATH'),
        collection_name=collection_name)


# Get relevant documents by query against law or investigation report
def get_relevant_documents_by_query(
        query: str,
        target_name: str = 'law'):
    # check if query is empty or string
    if not isinstance(query, str):
        logger.error(f'Query is not a string: {query}')
        return "Please provide a query."

    indexer = get_indexer(target_name)

    # similarity search
    # search_results = indexer.similarity_search(query)

    # get relevant documents by retriever
    # search_results = indexer.as_retriever().get_relevant_documents(query)

    # get relevant documents by multiquery retriever
    search_results = indexer.as_multiquery_retriever().get_relevant_documents(query)

    return search_results


# Function to get relevant documents by a website
def get_relevant_documents_by_website(site_link: str):
    from query import web, summary
    """
    Get relevant documents by a website.
    """
    if not site_link:
        return "Please provide a site link."

    # crawl a site and load all text from HTML webpages into a document format
    documents = web.crawler(site_link)

    # summarize documents
    summary_text = summary.summarization(documents)

    return get_relevant_documents_by_query(query=summary_text)


# QA against law or investigation report
def retrieval_qa(
        query: str,
        target_name: str = 'law'):
    from query import qa
    from util.openai import chatter

    # check if query is empty or string
    if not isinstance(query, str):
        logger.error(f'Query is not a string: {query}')
        return "Please provide a query."

    indexer = get_indexer(target_name=target_name)

    # determin chain type by target name
    chain_type = 'stuff' if target_name == 'law' else 'map_reduce'

    # create retrieval qa
    rqa = qa.EmbeddingsRetrievalQA(
        llm=chatter(),
        chain_type=chain_type,
        retriever=indexer.as_multiquery_retriever(),
        return_source_documents=True)

    search_results = rqa.query({"query": query})

    return search_results


# Run the cli app with arguments
if __name__ == '__main__':
    # display type of AI will be used for the app
    logger.info(f'Using AI type: {os.environ.get("OPENAI_API_TYPE")}')

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform-law-n-order',
                        action='store_true',
                        help='transform law and order data for creating embeddings')
    parser.add_argument('--create-law-embeddings',
                        action='store_true',
                        help='create law embeddings')
    parser.add_argument('--create-order-embeddings',
                        action='store_true',
                        help='create order embeddings')
    parser.add_argument('--create-investigation-embeddings',
                        action='store_true',
                        help='create investigation report embeddings')
    # get relevant documents by query
    parser.add_argument('--target-name',
                        type=str,
                        choices=['law', 'investigation'],
                        default='law',
                        help='query target name')
    parser.add_argument('--query', type=str, help='query string')
    parser.add_argument('--qa', type=str, help='query string by Retrieval QA')
    # get html text from a website
    parser.add_argument('--crawler', type=str, help='crawl a website')

    args = parser.parse_args()

    if args.transform_law_n_order:
        transform_law()
        transform_order()
    if args.create_law_embeddings:
        create_law_embeddings()
    if args.create_order_embeddings:
        create_order_embeddings()
    if args.create_investigation_embeddings:
        create_investigation_embeddings()
    if args.query:
        search_results = get_relevant_documents_by_query(
            query=args.query,
            target_name=args.target_name,)
        print('\n===== Relevant documents =====\n')
        print(search_results)
    if args.qa:
        search_results = retrieval_qa(
            query=args.qa,
            target_name=args.target_name,)
        print(search_results)
    if args.crawler:
        search_results = get_relevant_documents_by_website(args.crawler)
        print(search_results)
