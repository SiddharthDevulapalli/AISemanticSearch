# AISemanticSearch

The main objective of search engines is to find relevant documents based on a given query. However, this task is challenging due to potential differences in vocabulary between the query and the documents, as well as the presence of query words in irrelevant documents. To address this, we explore the use of neural word embeddings as a valuable source of evidence in document ranking.

This approach involves training a word2vec embedding model on a large unlabeled lyric corpus. During the ranking process, we map the words from the query into the input space of the model and the words from the documents into the output space. By computing cosine similarities between all query-document word pairs and aggregating the scores, we derive a measure of relevance between the query and the document.

For this project, I used lyrics data from genius.com.

The idea is to build a song search engine where a query is given and the most relevant songs are retrieved.

For this purpose, I used a Dual Embedding Space Model (DESM) based Retrieval system. In a DESM, word2vec embeddings are used to represent each word. Using these embeddings, a document vector is computed for every document.

Document Vector here would be the average of all the word embeddings of the document. 

For identifying relevant documents, each word of the query is compared with the document vector and similarity is calculated. The average of this similarity gives a score for every query-document pair. Using these scores, documents are ranked for each query and the top ranked documents are retrieved.

I used a sample dataset but can be scaled up using a vector database. Due to time constraints, I am just showing the sample search here.
I put lesser focus on UI and hence it is a very basic UI but it definitely does the expected job.

Future Scope: I plan to improve this by adding better UI, larger data put into a vector database, and and ensemble retrieval system for better ranking.


-Siddharth Devulapalli

