# Approach Explanation: Persona-Driven Document Intelligence

This document outlines the methodology and design decisions for the Persona-Driven Document Intelligence system. The system is designed to analyze a collection of PDF documents and extract the most relevant sections based on a given persona and job-to-be-done.

## 1. System Architecture

The system is built using a modular architecture, with distinct components for each stage of the analysis pipeline:

- **Input Processing:** Reads the input JSON file, which contains the persona, job-to-be-done, and document collection information.
- **Persona Analysis:** Analyzes the persona description to extract key characteristics such as role, domain, expertise level, and keywords.
- **Job Analysis:** Parses the job-to-be-done to identify actionable requirements, constraints, and deliverables.
- **Relevance Scoring:** Scores each section of the documents based on its relevance to the persona and job. This is the core of the system.
- **Content Refinement:** Cleans and summarizes the content of the most relevant sections.
- **Output Generation:** Generates the final output in the specified JSON format.

## 2. Core Components

### 2.1. Persona and Job Analysis

The `PersonaAnalyzer` and `JobParser` classes are responsible for understanding the user's context. They use a combination of keyword matching, pattern recognition, and NLP techniques to extract structured information from the raw text descriptions. This allows the system to tailor the analysis to the specific needs of the user.

### 2.2. Relevance Scoring

The `RelevanceScorer` is the heart of the system. It uses a multi-factor approach to score the relevance of each document section:

- **Semantic Similarity:** A lightweight sentence-transformer model (`all-MiniLM-L6-v2`) is used to calculate the semantic similarity between the user's query (a combination of persona and job keywords) and the text of each section. This allows the system to understand the meaning of the text, rather than just matching keywords.
- **Domain-Specific Boosting:** The system includes a dynamic boost/penalty system that adjusts the relevance scores based on domain-specific knowledge. For example, in the travel domain, sections that mention "group travel" or "budget" are given a higher score if the user is a travel planner organizing a group trip on a budget.
- **Keyword Matching:** Although semantic similarity is the primary factor, keyword matching is still used as a secondary signal to fine-tune the scores.

## 3. Performance and Constraints

The system is designed to meet the specified performance constraints:

- **CPU-Only:** The entire pipeline runs on a CPU. The sentence-transformer model is lightweight and optimized for CPU inference.
- **Model Size:** The `all-MiniLM-L6-v2` model is only 80MB, which is well within the 1GB limit.
- **Processing Time:** The use of a lightweight model and efficient scoring algorithms ensures that the processing time is within the 60-second limit for a typical document collection.
- **Offline Execution:** The system is containerized using Docker. The Dockerfile ensures that all dependencies, including the sentence-transformer model and NLTK data, are pre-installed in the container. This allows the system to run without any internet access.

## 4. Conclusion

This persona-driven document intelligence system provides a powerful and flexible solution for extracting relevant information from large document collections. By combining semantic understanding with domain-specific knowledge, the system is able to deliver highly accurate and personalized results that meet the specific needs of the user.
