# Deep Learning - Task 2
## Roland Gulbinovič - 2416108

The second task requires:
* realization of a semantic search tool for the selected context using a natural language model-based transformer neural networks to be used as a feature extractor and save representations to vector database.

## Data Gathering - web-scraping

web-scraping website - _Delfi.lt_.

I scraped all articles from delfi.lt/en/business - which are just articles about business written in English.

## Transformer

For the transformer I used - `all-MiniLM-L6-v2`.

This was chosen just because this is a small and lightweight transformer that is works well out-of-the-box for semantic search. It turns a sentence into a fixed-size vector (embedding) that captures its semantic meaning.

## Vector Database

The embeddings that we get from our transformer are then put into a `FAISS` vector database. This is important because from this database we can quickly retrieve similar articles for any new query. 

For the index I used `IndexFlatL2` - this uses L2 euclidean distance to compare vectors.

## Tokenizer and Language Model
Now we have a way of retrieving relevant articles for our new query, so all we need is to generate an answer. For this obviously we will need some kind of language model.
I chose a random open source LLM from https://github.com/eugeneyan/open-llms. The specific one that I chose is - `flan-t5-base`. This is an improved version of the `t5` model. It's trained to understand instructions and generate answers from context.

P.S. __Both__ the tokenizer and LLM were taken from `flan-t5-base`


## The overall pipeline

- We provide some kind of query. 
- The query gets embedded. It retrieves similar articles.
- The retrieved articles get used as context for the pre-trained LLM to generate an answer
