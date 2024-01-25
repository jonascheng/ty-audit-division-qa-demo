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
    from embeddings.law import transformer

    transformer(
        os.environ.get('LAW_FILEPATH_TRANSFORMED'),
        os.environ.get('LAW_FILEPATH_EMBEDDINGS'))


# Function to transform order embeddings
def transform_order_embeddings():
    from embeddings.law import transformer

    transformer(
        os.environ.get('ORDER_FILEPATH_TRANSFORMED'),
        os.environ.get('ORDER_FILEPATH_EMBEDDINGS'),
        collection_name_prefix='order',
        collection_partition_size=4,
        chunk_size=800,
        chunk_overlap=100)


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
    args = parser.parse_args()

    if args.transform_law_n_order:
        transform_law()
        transform_order()
    if args.transform_law_embeddings:
        transform_law_embeddings()
    if args.transform_order_embeddings:
        transform_order_embeddings()
