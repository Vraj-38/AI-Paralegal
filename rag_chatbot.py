import os
import time
import re
from collections import Counter
import numpy as np
from dotenv import load_dotenv
import pinecone
import requests
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import glob

# Load environment variables
load_dotenv()

class RAGChatbot:
    def __init__(self):
        # Initialize API keys from environment
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Initialize Pinecone
        self.pc = pinecone.Pinecone(
            api_key=self.pinecone_api_key,
            environment='us-east1'
        )
        
        # Initialize Gemini API settings
        self.gemini_model = "gemini-1.5-flash"
        self.gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
        
        # Initialize embedding model - using the same as in embeddings.py
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            api_key=self.google_api_key
        )
        
        # Initialize Pinecone index
        self.index_name = "ipd"
        self.index = self.pc.Index(self.index_name)
        
        # Vector dimension for dummy vectors
        self.vector_dimension = 768  # Standard for embedding-004
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )

        # Define general conversation patterns
        self.general_patterns = {
            "greeting": ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"],
            "how_are_you": ["how are you?", "how are you doing?", "how's it going?", "what's up?"],
            "what_do_you_do": ["what do you do?", "what are you?", "who are you?", "what is your purpose?", "what can you do?"],
            "goodbye": ["bye", "goodbye", "see you", "farewell", "take care"]
        }
        
        # Define stopwords for keyword extraction
        self.stopwords = {"a", "an", "the", "and", "or", "but", "if", "then", "else", "when", 
                    "at", "by", "for", "with", "about", "against", "between", "into", 
                    "through", "during", "before", "after", "above", "below", "to", "from", 
                    "up", "down", "in", "out", "on", "off", "over", "under", "again", 
                    "further", "then", "once", "here", "there", "all", "any", "both", 
                    "each", "few", "more", "most", "other", "some", "such", "no", "nor", 
                    "not", "only", "own", "same", "so", "than", "too", "very", "can", 
                    "will", "just", "should", "now", "what", "which", "how", "where", "is", "are"}

    def is_general_query(self, query: str) -> tuple[bool, str]:
        """
        Check if the query is a general conversation query.
        Now matches exact phrases or greetings only.
        """
        query_lower = query.lower().strip().rstrip('?!.,')
        
        # First, check for exact matches
        for category, patterns in self.general_patterns.items():
            if query_lower in patterns:
                return True, category
            
        # For greetings and goodbyes, also check if query contains only these words
        if category in ["greeting", "goodbye"]:
            query_words = set(query_lower.split())
            for pattern in patterns:
                pattern_words = set(pattern.split())
                if query_words == pattern_words:
                    return True, category
        
        # Special case for longer queries
        if len(query_lower.split()) > 3:  # If query is longer than 3 words
            return False, ""  # Assume it's a specific question
            
        return False, ""

    def get_general_response(self, category: str) -> str:
        """Get response for general conversation queries."""
        responses = {
            "greeting": "Hello! I'm your AI Paralegal assistant. How can I help you today?",
            "how_are_you": "I'm functioning well and ready to assist you with legal research and document analysis. How can I help you?",
            "what_do_you_do": "I am an AI Paralegal assistant specialized in analyzing legal documents and precedents. I can help you understand legal documents, extract relevant information from case files, and answer questions based on legal precedents.",
            "goodbye": "Goodbye! Feel free to return if you need any assistance with legal document analysis."
        }
        return responses.get(category, "I'm here to help you analyze legal documents and answer your questions.")

    def list_namespaces(self):
        """List all available namespaces in the index"""
        print("\n=== AVAILABLE NAMESPACES ===")
        try:
            stats = self.index.describe_index_stats()
            namespaces = stats.namespaces
            if namespaces:
                try:
                    print("Found " + str(len(namespaces)) + " namespaces:")
                    for ns, count in namespaces.items():
                        try:
                            print("- " + str(ns) + ": " + str(count) + " vectors")
                        except UnicodeEncodeError:
                            print("- [namespace with special characters]")
                except UnicodeEncodeError:
                    print("Found namespaces (count display error)")
                return list(namespaces.keys())
            else:
                print("No namespaces found in the index.")
                return []
        except Exception as e:
            try:
                print("Error listing namespaces: " + str(e))
            except UnicodeEncodeError:
                print("Error listing namespaces (display error)")
            return []

    def extract_keywords(self, text):
        """Extract important keywords from text for query expansion"""
        # Tokenize and filter words
        words = text.lower().split()
        important_words = [word for word in words if word not in self.stopwords and len(word) > 2]
        
        # Get most common words that might be important
        word_counts = Counter(important_words)
        keywords = [word for word, count in word_counts.most_common(5)]
        
        # Always include original words
        original_words = [w for w in words if len(w) > 2 and w not in self.stopwords]
        
        # Combine original query words with common keywords
        all_keywords = list(set(original_words + keywords))
        
        return all_keywords

    def expand_query(self, question):
        """Generate multiple query variations to improve retrieval"""
        # For very long queries (more than 10 words), limit expansion
        words = question.split()
        if len(words) > 10:
            # For long queries, just use the original query and extract key phrases
            print("Long query detected - using limited expansion")
            
            # Extract important keywords
            keywords = self.extract_keywords(question)
            
            # Original query is always included
            queries = [question]
            
            # Add one keyword-only query with the most important terms
            if len(keywords) > 3:
                keyword_query = " ".join(keywords[:5])  # Use only top 5 keywords
                queries.append(keyword_query)
            
            # Only print if there are expansions
            if len(queries) > 1:
                try:
                    print("Query expansion: " + queries[1])
                except UnicodeEncodeError:
                    print("Query expansion: [contains special characters]")
            
            return queries
        
        # For shorter queries, use more expansions but still limit them
        keywords = self.extract_keywords(question)
        
        # Original query is always included
        queries = [question]
        
        # Add keyword-only query
        if len(keywords) > 0:
            keyword_query = " ".join(keywords)
            if keyword_query != question.lower() and keyword_query not in queries:
                queries.append(keyword_query)
        
        # Only add question word removal for short queries (less than 6 words)
        if len(words) < 6:
            # Remove question words
            question_lower = question.lower()
            for q_word in ["what", "where", "when", "how", "why", "who", "is", "are", "can", "do", "does"]:
                if question_lower.startswith(q_word):
                    clean_q = question_lower.replace(q_word, "", 1).strip()
                    if clean_q and clean_q not in queries:
                        queries.append(clean_q)
                    break  # Only remove one question word
        
        # Only create bigrams if we have a reasonable number of keywords
        if 2 <= len(keywords) <= 5:
            # Create key bigrams (but limit to 2)
            count = 0
            for i in range(min(3, len(keywords))):
                for j in range(i+1, min(4, len(keywords))):
                    bigram = f"{keywords[i]} {keywords[j]}"
                    if bigram not in queries:
                        queries.append(bigram)
                        count += 1
                        if count >= 2:  # Only add up to 2 bigrams
                            break
                if count >= 2:
                    break
        
        # Limit to maximum 5 queries total
        queries = queries[:5]
        
        # Only print a simplified summary
        if len(queries) > 1:
            print(f"Query expansions ({len(queries)}): {queries[1]}" + 
                  (f", {queries[2]}" if len(queries) > 2 else ""))
        
        return queries

    def get_all_vectors(self, namespace, batch_size=3000):
        """Retrieve all vectors from a namespace using pagination."""
        all_matches = []
        
        # Use a dummy vector for query
        dummy_vector = [0.0] * self.vector_dimension
        
        # Pagination with ID filtering
        next_page_id = None
        while True:
            query_params = {
                "vector": dummy_vector,
                "namespace": namespace,
                "top_k": batch_size,
                "include_metadata": True
            }
            
            # Add ID filter for pagination
            if next_page_id:
                query_params["id"] = {"not_in": [next_page_id]}
            
            result = self.index.query(**query_params)
            
            if not result.matches:
                break
                
            all_matches.extend(result.matches)
            
            if len(result.matches) < batch_size:
                break
                
            # Get the last ID for pagination
            next_page_id = result.matches[-1].id
        
        return all_matches

    def keyword_search_in_namespace(self, query_terms, namespace, top_k=10):
        """Search for multiple keywords in a namespace"""
        
        # Get all vectors from the namespace
        matches = self.get_all_vectors(namespace)
        found_results = []
        
        # For each match, check for keyword presence
        for match in matches:
            if not match.metadata or 'text' not in match.metadata:
                continue
                
            text = match.metadata['text']
            text_lower = text.lower()
            
            # Check how many query terms are in the text
            matching_terms = [term for term in query_terms if term.lower() in text_lower]
            
            # Calculate score based on term presence
            if matching_terms:
                score = len(matching_terms) / len(query_terms)
                
                # Extract context around matches
                contexts = []
                for term in matching_terms:
                    term_lower = term.lower()
                    # Find all occurrences
                    start_idx = 0
                    while True:
                        idx = text_lower.find(term_lower, start_idx)
                        if idx == -1:
                            break
                        
                        start_context = max(0, idx - 200)
                        end_context = min(len(text), idx + len(term) + 200)
                        context = text[start_context:end_context]
                        contexts.append(context)
                        
                        # Move past this occurrence
                        start_idx = idx + len(term)
                
                # Deduplicate contexts
                unique_contexts = []
                for ctx in contexts:
                    if ctx not in unique_contexts:
                        unique_contexts.append(ctx)
                
                # Add to results
                found_results.append({
                    "id": match.id,
                    "namespace": namespace,
                    "score": score,
                    "matching_terms": matching_terms,
                    "text": text,
                    "contexts": unique_contexts,
                    "source": match.metadata.get("source", "Unknown")
                })
        
        # Sort by score
        found_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top results
        return found_results[:top_k]

    def vector_search(self, query, namespaces=None, top_k=10):
        """Perform vector search using embedding model"""
        # Generate embedding for the query
        query_embedding = self.embedding_model.embed_query(query)
        
        all_results = []
        
        # If namespaces not specified, search in all namespaces
        if not namespaces:
            namespaces = self.list_namespaces()
        
        # Search each namespace
        for namespace in namespaces:
            try:
                # Query Pinecone index
                query_results = self.index.query(
                    vector=query_embedding,
                    namespace=namespace,
                    top_k=top_k,
                    include_metadata=True
                )
                
                # Process results
                for match in query_results.matches:
                    if match.score > 0:  # Only include non-zero scores
                        if match.metadata and 'text' in match.metadata:
                            # Create context from text
                            text = match.metadata['text']
                            # Limit context to a manageable size
                            if len(text) > 800:
                                context = text[:800] + "..."
                            else:
                                context = text
                                
                            # Add to results
                            all_results.append({
                                "id": match.id,
                                "namespace": namespace,
                                "score": match.score,
                                "matching_terms": ["semantic match"],  # No specific keywords for vector search
                                "text": text,
                                "contexts": [context],
                                "source": match.metadata.get("source", "Unknown"),
                                "match_type": "vector"
                            })
            except Exception as e:
                print(f"Error in vector search for namespace {namespace}: {str(e)}")
        
        # Sort by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        return all_results[:top_k]

    def retrieve_context(self, question, top_k=10):
        """Advanced multi-strategy retrieval"""
        # Fix encoding issues by handling the output safely
        try:
            print("\nProcessing question: " + question)
        except UnicodeEncodeError:
            print("\nProcessing question: [contains special characters]")
        
        # 1. Expand the query for better retrieval
        expanded_queries = self.expand_query(question)
        
        # 2. Get all keywords for keyword search
        all_keywords = []
        for query in expanded_queries:
            all_keywords.extend(query.split())
        # Deduplicate
        all_keywords = list(set(all_keywords))
        
        # 3. Get all namespaces
        namespaces = self.list_namespaces()
        if not namespaces:
            print("No namespaces found to search.")
            return []
        
        # 4. Search each namespace using keywords
        all_results = []
        for namespace in namespaces:
            namespace_results = self.keyword_search_in_namespace(all_keywords, namespace, top_k=top_k)
            all_results.extend(namespace_results)
        
        # 5. Also perform vector search for semantic matching
        vector_results = self.vector_search(question, namespaces, top_k=top_k//2)
        all_results.extend(vector_results)
        
        # 6. Sort all results by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # 7. Remove duplicates by ID
        unique_results = []
        seen_ids = set()
        for result in all_results:
            if result['id'] not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result['id'])
        
        # 8. Take top_k results
        top_results = unique_results[:top_k]
        
        # Print results summary safely
        try:
            print("\nFound " + str(len(top_results)) + " relevant documents:")
            for i, result in enumerate(top_results[:3]):  # Print only top 3 for brevity
                matching_terms = ", ".join(result['matching_terms'])
                print(str(i+1) + ". Score: " + str(round(result['score'], 2)) + 
                      " | Namespace: " + str(result['namespace']) + 
                      " | Matching: " + matching_terms)
        except UnicodeEncodeError:
            print("\nFound relevant documents (display error)")
        
        return top_results

    def is_summary_request(self, question):
        """Check if the user's question is asking for a summary"""
        question_lower = question.lower()
        summary_keywords = [
            "summarize", "summary", "summarize", "brief", "tl;dr", 
            "synopsis", "condense", "recap", "overview", "digest"
        ]
        
        for keyword in summary_keywords:
            if keyword in question_lower:
                return True
        
        summary_patterns = [
            r"can you (give|provide) (me )?(a |an )?(quick |brief )?(summary|overview)",
            r"(summarize|sum up|condense)",
            r"(in short|in brief|in a nutshell)",
        ]
        
        for pattern in summary_patterns:
            if re.search(pattern, question_lower):
                return True
        
        return False

    def get_summary_with_method(self, context_results, question, method="standard"):
        """Generate summary using different methods"""
        
        if not context_results:
            return "No relevant documents found to summarize."
        
        # Prepare context
        all_texts = []
        for result in context_results:
            all_texts.extend(result['contexts'])
        
        combined_text = "\n\n".join(all_texts)
        
        # Define different summarization instructions based on method
        summary_prompts = {
            "standard": f"""Summarize the following legal information in a comprehensive way:

{combined_text}

Provide a well-structured summary that covers the main legal points.""",

            "extractive": f"""Create an extractive summary of the following legal information by selecting and combining the most important sentences directly from the source text:

{combined_text}

Your summary should:
1. Only use sentences that appear verbatim in the source text
2. Maintain the original wording of selected sentences
3. Combine these extracted sentences into a coherent summary
4. Highlight the most important facts and legal statements""",

            "abstractive": f"""Create an abstractive summary of the following legal information by reformulating and synthesizing the content in your own words:

{combined_text}

Your summary should:
1. Restate information in new language not found in the original text
2. Consolidate legal ideas and concepts across the entire text
3. Use your own phrasing rather than copying sentences from the source
4. Maintain the core legal meaning while using fresh language""",

            "hybrid": f"""Create a hybrid summary of the following legal information by combining both extractive and abstractive techniques:

{combined_text}

Your summary should:
1. Use some direct quotes or key sentences from the source when they are particularly important legal points
2. Rephrase and synthesize other content in your own words
3. Blend extracted content and reformulated content seamlessly
4. Ensure comprehensive coverage of the main legal points""",

            "query-focused": f"""Create a focused summary of the following legal information that specifically addresses this query: "{question}"

{combined_text}

Your summary should:
1. Only include information relevant to answering the query
2. Prioritize content that directly relates to what was asked
3. Omit details that don't help address the specific legal question
4. Be targeted and precise in addressing exactly what was asked"""
        }
        
        # Get the appropriate prompt
        prompt = summary_prompts[method]
        
        # Use the call_gemini_api method we already have
        return self.call_gemini_api(prompt)

    def call_gemini_api(self, prompt, max_tokens=2000, temperature=0.0):
        """Call the Gemini API with the given prompt"""
        url_with_key = f"{self.gemini_url}?key={self.google_api_key}"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.95,
                "topK": 64
            }
        }
        
        try:
            response = requests.post(url_with_key, headers=headers, json=payload)
            response.raise_for_status()
            
            response_json = response.json()
            
            # Extract text from Gemini response format
            if 'candidates' in response_json and len(response_json['candidates']) > 0:
                if 'content' in response_json['candidates'][0]:
                    content = response_json['candidates'][0]['content']
                    if 'parts' in content and len(content['parts']) > 0:
                        return content['parts'][0]['text']
            
            return "Error: Unable to extract text from Gemini API response."
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                return "Error: Invalid Gemini API key. Please check your credentials."
            else:
                error_message = response.text
                try:
                    error_json = json.loads(error_message)
                    if 'error' in error_json and 'message' in error_json['error']:
                        error_message = error_json['error']['message']
                except:
                    pass
                return f"Error from Gemini API: {error_message}"

    def chat_with_gemini(self, context_results, question):
        """Enhanced chat completion with better context formatting"""
        if not context_results:
            context_str = "No relevant documents found in the knowledge base."
            sources_str = ""
        else:
            # Format context chunks with source information
            context_chunks = []
            sources = set()
            
            for i, result in enumerate(context_results):
                source = result['source']
                namespace = result['namespace']
                matching_terms = ", ".join(result['matching_terms'])
                sources.add(f"{source} (from {namespace})")
                
                # Add each context with formatting
                for j, context in enumerate(result['contexts']):
                    if context:  # Check if context is not empty
                        context_chunks.append(f"[Document {i+1}, Excerpt {j+1}, Keywords: {matching_terms}] {context}")
            
            context_str = "\n\n".join(context_chunks)
            sources_str = "Sources:\n" + "\n".join(f"- {s}" for s in sources)
        
        # Enhanced prompt with more specific instructions for legal assistant formatting
        prompt = f"""You are an AI Paralegal assistant specialized in analyzing legal documents and precedents. Answer the user's question directly and comprehensively using the retrieved information below.

User Question: {question}

Retrieved Information:
{context_str}

Instructions:
1. Base your answer solely on the provided information.
2. Provide detailed, accurate responses that directly address the legal question.
3. Present legal data confidently without citing document numbers.
4. Use structured, properly indented paragraphs without excessive line breaks or whitespace.
5. Format in a professional legal manner, using numbered lists only when appropriate for legal steps, arguments, or citations.
6. Use professional legal terminology and formal language typical of legal documents.
7. Maintain formal tone and precise legal phrasing throughout.
8. Avoid using asterisks, bullet points, markdown formatting, or other non-standard characters.
9. Ensure all content is properly structured with logical paragraph breaks (not excessive spacing).
10. Acknowledge when information is incomplete rather than making assumptions about legal matters.
11. Do not include the preamble "here's what I found" or similar introductory phrases.
12. Present your answer as a properly formatted legal analysis or opinion that could be used in a professional legal setting.

Answer:"""
        
        # Call Gemini API
        return self.call_gemini_api(prompt)

    def chat(self, query: str):
        """Process a user query and return a response"""
            # Check if it's a general conversation query first
        is_general, category = self.is_general_query(query)
        if is_general:
            return {
                "answer": self.get_general_response(category),
                "sources": [],
                "contexts": []
            }

        # Check if this is a summary request
        if self.is_summary_request(query):
            try:
                print("\nDetected summary request. Using standard summary method.")
            except UnicodeEncodeError:
                print("\nDetected summary request.")

            # Get context using our advanced retrieval
            context_results = self.retrieve_context(query, top_k=10)

            # Generate a standard summary
            answer = self.get_summary_with_method(context_results, query, method="standard")

        # Format sources
            sources = []
            seen_sources = set()
            for result in context_results:
                source_key = f"{result['source']}:{result['namespace']}"
                if source_key not in seen_sources:
                    sources.append({
                        "file": result['source'],
                        "namespace": result['namespace']
                    })
                    seen_sources.add(source_key)

            # Collect contexts for RAGAS
            contexts = []
            for result in context_results:
                for context_piece in result.get("contexts", []):
                    contexts.append(context_piece)

            return {
                "answer": answer,
                "sources": sources,
                "contexts": contexts
            }

        else:
            # Use our improved retrieval for regular queries
            context_results = self.retrieve_context(query, top_k=5)

            if not context_results:
                return {
                    "answer": "I couldn't find any relevant information in the available documents. Could you please rephrase your question or provide more context?",
                    "sources": [],
                    "contexts": []
                }

            # Get answer using enhanced chat function
            answer = self.chat_with_gemini(context_results, query)

            # Format sources
            sources = []
            seen_sources = set()
            for result in context_results:
                source_key = f"{result['source']}:{result['namespace']}"
                if source_key not in seen_sources:
                    try:
                        page = int(result['id'].split('-pdf-')[1].split('-')[0]) if '-pdf-' in result['id'] else 1
                        sources.append({
                            "file": result['source'],
                            "page": page
                        })
                        seen_sources.add(source_key)
                    except:
                        sources.append({
                            "file": result['source'],
                            "page": 1
                        })
                        seen_sources.add(source_key)

            # Collect contexts for RAGAS
            contexts = []
            for result in context_results:
                for context_piece in result.get("contexts", []):
                    contexts.append(context_piece)

            return {
                "answer": answer,
                "sources": sources,
                "contexts": contexts
            }


def main():
    """Main entry point for the chatbot"""
    # Initialize the chatbot
    chatbot = RAGChatbot()
    
    # Check for available namespaces
    available_namespaces = chatbot.list_namespaces()
    
    if not available_namespaces:
        print("\n⚠️ Warning: No document namespaces found in the Pinecone index.")
        print("Please make sure documents have been properly indexed using the embedding script.")
        print("The chatbot will continue but may not find relevant answers without indexed documents.")
    else:
        try:
            print("\n✓ Found " + str(len(available_namespaces)) + " document namespaces in the index.")
        except UnicodeEncodeError:
            print("\n✓ Found document namespaces in the index.")
    
    # Chat loop
    print("\n🤖 AI Paralegal Chatbot Ready! Type 'quit' to exit.")
    print("This chatbot uses advanced retrieval to find relevant legal information.")
    
    while True:
        query = input("\nYou: ")
        if query.lower() in ['quit', 'exit']:
            print("Goodbye! Thank you for using the AI Paralegal assistant.")
            break
            
        if not query.strip():
            print("Please enter a question.")
            continue
            
        try:
            # Measure retrieval and response time
            start_time = time.time()
            response = chatbot.chat(query)
            total_time = time.time() - start_time
            
            print("\nAI Paralegal:", response["answer"])
            
            if response["sources"]:
                print("\nSources:")
                for source in response["sources"]:
                    try:
                        if "page" in source:
                            print("- " + str(source['file']) + ", Page " + str(source['page']))
                        else:
                            print("- " + str(source['file']))
                    except UnicodeEncodeError:
                        print("- [Source with special characters]")
            
            try:
                print("\n(Processing time: " + str(round(total_time, 2)) + "s)")
            except UnicodeEncodeError:
                print("\n(Processing time calculation error)")
        except Exception as e:
            try:
                print("\n❌ Error: " + str(e))
            except UnicodeEncodeError:
                print("\n❌ Error occurred (details contain special characters)")
            print("Please try again with a different query or check your API keys and network connection.")

if __name__ == "__main__":
    main()
