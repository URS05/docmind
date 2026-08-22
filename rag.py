"""
DocMind - Intelligent Knowledge Retrieval and Generation System

Author : Umair Rahman Shaik
GitHub : https://github.com/URS05

DocMind is an advanced Retrieval-Augmented Generation (RAG) system that
combines dense vector search with large language model reasoning to deliver
accurate, context-grounded answers over any custom document corpus.

Core Modules:
    - AutoId          : Utility for detecting txtai-generated auto IDs (UUID/numeric)
    - GraphContext    : Builds semantic knowledge graph contexts for Graph RAG queries
    - Application     : Primary RAG engine that manages LLM, embeddings, ingestion, and the
                        Streamlit chat interface
"""

import os
import re

from glob import glob
from io import BytesIO
from uuid import UUID

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from txtai import Embeddings, LLM, RAG
from txtai.pipeline import Textractor

# Initialize module-level logger via Streamlit's logging interface
logger = st.logger.get_logger(__name__)


class AutoId:
    """
    Utility class for detecting txtai-generated automatic identifiers.

    txtai assigns either UUID v5 strings or monotonically incrementing
    integers as auto IDs when documents are indexed without explicit IDs.
    This class provides a single static method to identify such IDs so
    that graph nodes can be labelled with human-readable topic names
    instead of raw identifiers.
    """

    @staticmethod
    def valid(uid):
        """
        Determine whether a given identifier is a txtai-generated auto ID.

        An auto ID is defined as either:
            - A UUID v5 string (assigned when autoid='uuid5' is configured)
            - A plain integer or digit-only string (sequential numeric ID)

        Args:
            uid: The identifier to evaluate. May be a str, int, or UUID object.

        Returns:
            True  - if uid is a UUID string or a numeric value
            False - if uid is a user-assigned human-readable string key
        """

        # Check if this is a UUID
        try:
            return UUID(str(uid))
        except ValueError:
            pass

        # Return True if this is numeric, False otherwise
        return isinstance(uid, int) or uid.isdigit()


class GraphContext:
    """
    Graph-augmented context builder for DocMind's Graph RAG pipeline.

    This class intercepts user queries and determines whether they are
    targeting the knowledge graph layer of the embeddings index. When a
    graph query is detected, it executes a Cypher-based path traversal
    over the semantic graph, assembles a rich contextual node set, renders
    a visual graph diagram, and returns a structured context list for the LLM.

    Supported query patterns:
        - 'gq: <natural language>'         Graph RAG with query expansion
        - 'concept_a -> concept_b'         Semantic path traversal
        - 'concept_a -> concept_b gq: ...' Path traversal + targeted query
    """

    def __init__(self, embeddings, context):
        """
        Initialise a new GraphContext for a given query session.

        Args:
            embeddings (Embeddings): The txtai Embeddings instance that holds
                                    the vector store and knowledge graph.
            context    (int)       : Maximum number of graph nodes to include
                                    in the assembled LLM context.
        """

        self.embeddings = embeddings
        self.context = context

    def __call__(self, question):
        """
        Attempts to create a graph context for the input question. This method checks if:
          - Embeddings has a graph
          - Question is a graph query

        If both of the above are true, the graph is scanned to find the best matching records
        to use as a context.

        Args:
            question: input question

        Returns:
            question, [context]
        """

        query, concepts, context = self.parse(question)
        if self.embeddings.graph and (query or concepts):
            # Generate graph path query
            path = self.path(query, concepts)

            # Build graph network from path query
            graph = self.embeddings.graph.search(path, graph=True)
            if graph.count():
                # Draw and display graph
                response = self.plot(graph)
                st.write(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

                # Build graph context
                context = [
                    {
                        "id": graph.attribute(node, "id"),
                        "text": graph.attribute(node, "text"),
                    }
                    for node in list(graph.scan())
                ]
                if context:
                    # Default prompt
                    default = (
                        "Write a title and text summarizing the context.\n"
                        f"Include the following concepts: {concepts} if they're mentioned in the context."
                    )

                    # Set question to query if available, otherwise use default prompt
                    question = query if query else default

        return question, context

    def parse(self, question):
        """
        Parse a user question to detect and extract graph query directives.

        A query is classified as a graph query when it contains:
            - Arrow notation ('->') indicating a concept path traversal
            - The 'gq: ' prefix indicating a graph-augmented natural language query

        Args:
            question (str): Raw user input string from the chat interface.

        Returns:
            tuple: (query, concepts, context) where
                query    - The natural language sub-query (str or None)
                concepts - Ordered list of concept strings for path traversal
                context  - Always None at parse time; populated downstream
        """

        # Graph query prefix
        prefix = "gq: "

        # Parse graph query
        query, concepts, context = None, None, None
        if "->" in question or question.strip().lower().startswith(prefix):
            # Split into concepts
            concepts = [x.strip() for x in question.strip().lower().split("->")]

            # Parse query out of concepts, if necessary
            if prefix in concepts[-1]:
                query, concepts = concepts[-1], concepts[:-1]

                # Look for search prefix
                query = [x.strip() for x in query.split(prefix, 1)]

                # Add concept, if necessary
                if query[0]:
                    concepts.append(query[0])

                # Extract query, if present
                if len(query) > 1:
                    query = query[1]

        return query, concepts, context

    def path(self, question, concepts):
        """
        Construct a Cypher MATCH PATH query from the parsed graph intent.

        Two resolution strategies are applied:
            - Concept list provided: For each concept string, the single best-matching
              embedding vector in the graph is located and used as a path node anchor.
            - No concepts (query only): The top-3 results from an embedding search
              on the question text are used as path node anchors.

        The resolved node IDs are joined with directed relationship wildcards
        (depth 1-4) and wrapped in a Cypher MATCH PATH statement.

        Args:
            question (str)       : Natural language sub-query (used when no concepts).
            concepts (list[str]) : Ordered list of concept strings for path anchoring.

        Returns:
            str: A Cypher MATCH PATH query string for execution on the graph backend.
        """

        # Find graph nodes
        ids = []
        if concepts:
            for concept in concepts:
                uid = self.embeddings.search(concept, 1)[0]["id"]
                ids.append(f'({{id: "{uid}"}})')
        else:
            for x in self.embeddings.search(question, 3):
                ids.append(f"({{id: \"{x['id']}\"}})")

        # Create graph path query
        ids = "-[*1..4]->".join(ids)
        query = f"MATCH P={ids} RETURN P LIMIT {self.context}"
        logger.debug(query)

        return query

    def plot(self, graph):
        """
        Render the traversed knowledge graph as a PNG image for display.

        Uses NetworkX spring layout with fixed seed for reproducible positioning.
        Nodes are coloured and labelled with their LLM-generated topic names.
        Duplicate nodes (topics with cosine similarity >= 0.9) are merged before
        plotting to reduce visual noise.

        Args:
            graph: A txtai Graph instance containing traversed nodes and edges.

        Returns:
            PIL.Image.Image: An in-memory PNG image of the rendered graph.
        """

        # Deduplicate and label graph
        graph, labels = self.deduplicate(graph, 0.9)

        options = {
            "node_size": 700,
            "node_color": "#ffbd45",
            "edge_color": "#e9ecef",
            "font_color": "#454545",
            "font_size": 10,
            "alpha": 1.0,
        }

        # Draw graph
        _, ax = plt.subplots(figsize=(9, 5))
        pos = nx.spring_layout(graph.backend, seed=0, k=0.9, iterations=50)
        nx.draw_networkx(graph.backend, pos=pos, labels=labels, **options)

        # Disable axes and draw margins
        ax.axis("off")
        plt.margins(x=0.15)

        # Save and return image
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        return Image.open(buffer)

    def deduplicate(self, graph, threshold):
        """
        Deduplicates input graph. This method merges nodes with topics having a similarity of more
        than the input threshold. This method also builds a dictionary of labels for each node.

        Args:
            graph: input graph
            threshold: topic merge threshold

        Returns:
            graph, labels
        """

        labels, topics, deletes = {}, {}, []
        for node in graph.scan():
            uid, topic = graph.attribute(node, "id"), graph.attribute(node, "topic")
            label = topic if AutoId.valid(uid) and topic else uid

            # Find similar topics
            topicnames = list(topics.keys())
            pid, pscore = (
                self.embeddings.similarity(label, topicnames)[0]
                if topicnames
                else (0, 0.0)
            )
            primary = topics[topicnames[pid]] if pscore >= threshold else None

            if not primary:
                # Set primary node
                labels[node], topics[label] = label, node
            else:
                # Copy edges to primary node
                logger.debug(f"DUPLICATE NODE: {label} - {topicnames[pid]}")
                edges = graph.edges(node)
                if edges:
                    for target, attributes in graph.edges(node).items():
                        if primary != target:
                            graph.addedge(primary, target, **attributes)

                # Add duplicate node to delete list
                deletes.append(node)

        # Delete duplicate nodes
        graph.delete(deletes)

        return graph, labels


class Application:
    """
    DocMind - Core RAG Application Engine.

    This class is the central controller for the DocMind system. It manages:

        - LLM initialisation and configuration (local or API-backed)
        - Embeddings index creation, loading, and dynamic updates
        - Document ingestion and text extraction pipeline (Textractor)
        - LLM-driven topic labelling for knowledge graph nodes
        - Prompt construction and RAG pipeline execution
        - Streamlit conversational UI rendering and session state management

    Author: Umair Rahman Shaik (https://github.com/URS05)
    """

    def __init__(self):
        """
        Initialise the DocMind RAG engine.

        Execution order:
            1. Initialise the Textractor pipeline (lazy - set to None until first use)
            2. Load or instantiate the LLM backend
            3. Load or create the Embeddings vector index + knowledge graph
            4. Configure the RAG pipeline with the system persona and prompt template
        """

        # Textractor is lazily initialised on first document ingestion call
        self.textractor = None

        # Initialise the LLM backend.
        # Defaults to Qwen2.5-0.5B (lightweight, CPU-friendly) for rapid prototyping.
        # Override by setting LLM= in your .env file to use any HuggingFace model,
        # llama.cpp GGUF path, or API model string (e.g. 'gpt-4o', 'ollama/llama3').
        self.llm = LLM(os.environ.get("LLM", "Qwen/Qwen2.5-0.5B-Instruct"))

        # Load embeddings
        self.embeddings = self.load()

        # Number of document chunks retrieved per query and injected into the LLM prompt
        self.context = int(os.environ.get("CONTEXT", 10))

        # Grounded RAG prompt template - constrains the LLM to answer strictly from
        # retrieved context, minimising hallucination and improving factual accuracy.
        template = """
Answer the following question using only the context below. Only include information
specifically discussed. Do not use any prior knowledge outside the provided context.

question: {question}
context: {context} """

        # Assemble the full RAG pipeline:
        #   - embeddings: supplies vector-retrieved or graph-retrieved context
        #   - llm:        generates the final grounded answer
        #   - system:     DocMind assistant persona injected as the system message
        #   - template:   structured prompt that enforces context-only reasoning
        self.rag = RAG(
            self.embeddings,
            self.llm,
            system=(
                "You are DocMind, an intelligent knowledge assistant built by Umair Rahman Shaik. "
                "You answer questions accurately and concisely, using only the retrieved context "
                "provided to you. If the answer is not contained in the context, say so clearly."
            ),
            template=template,
            context=self.context,
        )

    def load(self):
        """
        Creates or loads an Embeddings instance.

        Returns:
            Embeddings
        """

        embeddings = None

        # Raw data path
        data = os.environ.get("DATA")

        # Embeddings database path (defaults to empty index for instant startup; set EMBEDDINGS in .env for pre-built index)
        database = os.environ.get("EMBEDDINGS", "")

        # Check for existing index
        if database:
            logger.debug(f"LOAD INDEX: {database}")
            embeddings = Embeddings()
            if embeddings.exists(database):
                embeddings.load(database)
            elif not os.path.isabs(database) and embeddings.exists(
                cloud={"provider": "huggingface-hub", "container": database}
            ):
                embeddings.load(provider="huggingface-hub", container=database)
            else:
                logger.debug(f"NO INDEX FOUND: {database}")
                embeddings = None

        # Default embeddings index if not found
        embeddings = embeddings if embeddings else self.create()

        # Add content from data directory, if provided
        if data:
            logger.debug(f"INDEX DATA: {data}")
            embeddings.upsert(self.stream(data))

            # Create LLM-generated topics
            self.infertopics(embeddings, 0)

            # Save embeddings, if necessary
            self.persist(embeddings)

        return embeddings

    def addurl(self, url):
        """
        Adds content at URL to this embeddings index.

        Args:
            url: input url
        """

        # Store number in index before indexing
        start = self.embeddings.count()

        # Add file to embeddings index
        self.embeddings.upsert(self.extract(url))

        # Create LLM-generated topics
        self.infertopics(self.embeddings, start)

        # Save embeddings, if necessary
        self.persist(self.embeddings)

    def create(self):
        """
        Creates a new empty Embeddings index.

        Returns:
            Embeddings
        """

        # Create empty embeddings database
        return Embeddings(
            autoid="uuid5",
            path="intfloat/e5-large",
            instructions={"query": "query: ", "data": "passage: "},
            content=True,
            graph={"approximate": False, "minscore": 0.7},
        )

    def stream(self, data):
        """
        Runs a textractor pipeline and streams extracted content from a data directory.

        Args:
            data: input data directory
        """

        # Stream sections from content
        for sections in self.extract(glob(f"{data}/**/*", recursive=True)):
            yield from sections

    def extract(self, inputs):
        """
        Extract sections from inputs using a Textractor pipeline.

        Args:
            inputs: input content

        Returns:
            extracted content
        """

        # Initialize textractor
        if not self.textractor:
            self.textractor = Textractor(
                paragraphs=True,
                backend=os.environ.get("TEXTBACKEND", "available"),
                safeopen=os.environ.get("SAFEOPEN", "True").lower() in ("true", "1"),
            )

        # Extract text
        return self.textractor(inputs)

    def infertopics(self, embeddings, start):
        """
        Traverses the graph associated with an embeddings instance and adds
        LLM-generated topics for each entry.

        Args:
            embeddings: embeddings database
            start: number of records before indexing
        """

        if embeddings.graph:
            batch = []
            for uid in tqdm(
                embeddings.graph.scan(),
                desc="Inferring topics",
                total=embeddings.graph.count() - start,
            ):
                # Infer topic if id is an autoid and topic is empty
                rid = embeddings.graph.attribute(uid, "id")
                topic = embeddings.graph.attribute(uid, "topic")
                if AutoId.valid(rid) and not topic:
                    text = embeddings.graph.attribute(uid, "text")
                    text = text if text else rid

                    batch.append((uid, text))
                    if len(batch) == 32:
                        self.topics(embeddings, batch)
                        batch = []

            if batch:
                self.topics(embeddings, batch)

    def persist(self, embeddings):
        """
        Saves an embeddings index if the PERSIST parameter is set.

        Args:
            embeddings: embeddings to save
        """

        persist = os.environ.get("PERSIST")
        if persist:
            logger.debug(f"SAVE INDEX: {persist}")
            embeddings.save(persist)

    def topics(self, embeddings, batch):
        """
        Generates a batch of topics with a LLM. Topics are set directly on the embeddings
        instance.

        Args:
            embeddings: embeddings database
            batch: batch of (id, text) elements
        """

        prompt = """
Create a simple, concise topic for the following text. Only return the topic name.

Text:
{text}"""

        # Build batch of prompts
        prompts = []
        for uid, text in batch:
            text = text if re.search(r"\w+", text) else uid
            prompts.append([{"role": "user", "content": prompt.format(text=text)}])

        # Check if batch processing is enabled
        topicsbatch = os.environ.get("TOPICSBATCH")
        kwargs = {"batch_size": int(topicsbatch)} if topicsbatch else {}

        # Run prompt batch and set topics
        for x, topic in enumerate(
            self.llm(
                prompts,
                maxlength=int(os.environ.get("MAXLENGTH", 2048)),
                stripthink=os.environ.get("STRIPTHINK", "false").lower()
                in ("true", "1"),
                **kwargs,
            )
        ):
            # Set topic attribute
            uid = batch[x][0]
            embeddings.graph.addattribute(uid, "topic", topic)

            # Add topic to topics
            topics = embeddings.graph.topics
            if topics:
                if topic not in topics:
                    topics[topic] = []

                topics[topic].append(uid)

    def instructions(self):
        """
        Generate the initial welcome message displayed in the DocMind chat interface.

        The welcome message includes:
            - An introductory greeting from the DocMind assistant
            - Example queries tailored to the loaded index
            - Instructions for ingesting new documents into the knowledge base
            - Graph RAG usage guide (if the embeddings index has a graph enabled)

        Returns:
            str: Markdown-formatted welcome and instructions string.
        """

        # Example queries
        if "EXAMPLES" in os.environ:
            examples = [x.strip() for x in os.environ["EXAMPLES"].split(";")]
        else:
            examples = [
                "Who created Linux?",
                "gq: Tell me about Linux",
                "linux -> macos -> microsoft windows",
                "linux -> macos -> microsoft windows gq: Tell me about Linux",
            ]

        # Welcome message and base usage instructions
        instructions = (
            "Welcome to DocMind - your intelligent knowledge retrieval system.\n\n"
            f"Ask a question such as `{examples[0]}`\n\n"
            f"{'The knowledge base is currently empty. Add documents below to begin.' if not self.embeddings.count() else 'Knowledge base loaded and ready.'}\n\n"
            "**Add data to the knowledge base:**\n\n"
            "- `# https://example.com/paper.pdf` - Index a web URL or document\n"
            "- `# /path/to/your/file.pdf`         - Index a local file\n"
            "- `# Any custom text or notes here!` - Index raw text directly"
        )

        # Graph instructions
        if "graph" in self.embeddings.config:
            instructions += (
                "\n\nThis index also supports GraphRAG. Examples are shown below.\n"
                f"- `{examples[1]}`\n"
                "  - Graph RAG query, the `gq: ` prefix enables graph RAG\n"
                f"- `{examples[2]}`\n"
                "  - Graph path query for a list of concepts separated by `->`\n"
                "  - The graph path is analyzed and described by the LLM\n"
                f"- `{examples[3]}`\n"
                "  - Graph path with a graph RAG query"
            )

        return instructions

    def settings(self):
        """
        Generates a message with current settings.

        Returns:
            settings
        """

        # Generate config settings rows
        config = "\n".join(
            f"|{name}|{os.environ.get(name)}|"
            for name in ["EMBEDDINGS", "DATA", "PERSIST", "LLM"]
            if name
        )

        return (
            "The following is a table with the current settings.\n"
            f"|Name|Value|\n"
            f"|----|-----|\n"
            f"|RECORD COUNT|{self.embeddings.count()}|\n"
        ) + config

    def run(self):
        """
        Runs a Streamlit application.
        """

        if "messages" not in st.session_state.keys():
            # Add instructions
            st.session_state.messages = [
                {"role": "assistant", "content": self.instructions()}
            ]

        if question := st.chat_input("Your question"):
            message = question
            if question.startswith("#"):
                message = f"Upload request for _{message.split('#')[-1].strip()}_"

            st.session_state.messages.append({"role": "user", "content": message})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if (
            st.session_state.messages
            and st.session_state.messages[-1]["role"] != "assistant"
        ):
            with st.chat_message("assistant"):
                logger.debug(f"USER INPUT: {question}")

                # Check for file upload
                if question.startswith("#"):
                    url = question.split("#")[1].strip()
                    with st.spinner(f"Adding {url} to index"):
                        self.addurl(url)

                    response = f"Added _{url}_ to index"
                    st.write(response)

                # Show settings
                elif question == ":settings":
                    response = self.settings()
                    st.write(response)

                else:
                    # Check for Graph RAG
                    graph = GraphContext(self.embeddings, self.context)
                    question, context = graph(question)

                    # Graph RAG
                    if context:
                        logger.debug(
                            f"----------------- GRAPH CONTEXT ({len(context)})----------------"
                        )
                        for x in context:
                            logger.debug(x)

                        # Transform context into a list of text
                        context = [x["text"] for x in context]

                    # Vector RAG
                    else:
                        logger.debug("-----------------CONTEXT----------------")
                        for x in self.embeddings.search(question, self.context):
                            logger.debug(x)

                    # Run RAG
                    response = self.rag(
                        question,
                        context,
                        maxlength=int(os.environ.get("MAXLENGTH", 4096)),
                        stream=True,
                        stripthink=os.environ.get("STRIPTHINK", "False").lower()
                        in ("true", "1"),
                    )

                    # Render response
                    response = st.write_stream(response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )


@st.cache_resource(show_spinner="Initializing models and database...")
def create():
    """
    Creates and caches a Streamlit application.

    Returns:
        Application
    """

    return Application()


if __name__ == "__main__":
    # Disable HuggingFace tokenizer parallelism warnings in Streamlit's multi-threaded runtime
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Configure the Streamlit page - DocMind branding
    st.set_page_config(
        page_title="DocMind - Intelligent Knowledge Retrieval",
        page_icon="D",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items={
            "About": (
                "**DocMind** - Intelligent Knowledge Retrieval and Generation System\n\n"
                "Built by **Umair Rahman Shaik**  \n"
                "GitHub: [URS05](https://github.com/URS05)"
            )
        },
    )
    st.title(os.environ.get("TITLE", "DocMind"))
    st.caption("Intelligent Knowledge Retrieval and Generation | Built by Umair Rahman Shaik")

    # Instantiate and launch the DocMind RAG application
    app = create()
    app.run()
