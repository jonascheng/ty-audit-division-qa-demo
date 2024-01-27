# chunker function
# calculate batch size based on total documents
# batch size is the number of documents to be processed in one batch
def chunker(documents: list[any]) -> (list[list[any]], int):
    if len(documents) > 10000:
        batch_size = int(len(documents) / 10000)
    elif len(documents) > 1000:
        batch_size = int(len(documents) / 1000)
    elif len(documents) > 100:
        batch_size = int(len(documents) / 100)
    else:
        batch_size = len(documents)
    return chunk_by_batch_size(
        documents=documents,
        batch_size=batch_size), batch_size


def chunk_by_batch_size(documents: list[any], batch_size: int) -> list[list[any]]:
    batches = [
        documents[i:i + batch_size] for i in
        range(0, len(documents), batch_size)]
    return batches
