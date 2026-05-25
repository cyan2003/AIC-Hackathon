"""
Unit tests for document cleaning, chunking, validation, and metadata extraction.
"""
import pytest
from core.config import IngestionConfig, ChunkingConfig, ChunkingStrategy
from ingestion.interfaces import RawDocument
from ingestion.processors.cleaner import DefaultTextCleaner
from ingestion.processors.validator import DefaultDocumentValidator
from ingestion.processors.chunker import (
    FixedSizeChunker,
    SentenceChunker,
    ParagraphChunker,
    create_chunker,
    create_chunker_for_document_type,
)
from ingestion.processors.section_chunker import ResumeSectionChunker, JDSectionChunker
from ingestion.processors.resume_metadata_extractor import (
    ResumeMetadataExtractor,
    LLMResumeMetadataExtractor,
    LLMClient,
)
from ingestion.processors.jd_metadata_extractor import JDMetadataExtractor


def test_default_text_cleaner():
    """Test text cleaning options (whitespace, HTML, URLs, case)."""
    cleaner = DefaultTextCleaner(
        lowercase=True,
        strip_html=True,
        remove_urls=True,
        min_line_length=5,
    )
    
    dirty_text = (
        "<html><body>\n"
        "  Hello   World! \n"
        "Go to https://google.com for more info.\n"
        "Abc\n"  # under min_line_length (3 < 5)
        "Line 2 is long enough.\n"
        "</body></html>"
    )
    
    cleaned = cleaner.clean(dirty_text)
    assert "hello world!" in cleaned
    assert "google.com" not in cleaned
    assert "abc" not in cleaned
    assert "line 2 is long enough." in cleaned
    assert "<html>" not in cleaned


def test_default_document_validator():
    """Test size, extension, and mime-type validations."""
    config = IngestionConfig(
        max_document_size_mb=1,
        supported_extensions=[".pdf", ".docx"],
    )
    validator = DefaultDocumentValidator(config)

    # Valid
    doc_valid = RawDocument(
        source="john_doe.pdf",
        content="This is a valid test file.",
        mime_type="application/pdf",
    )
    assert validator.validate(doc_valid) is True

    # Empty content
    doc_empty = RawDocument(source="empty.pdf", content="", mime_type="application/pdf")
    assert validator.validate(doc_empty) is False
    assert "Document content is empty." in validator.validation_errors(doc_empty)

    # Oversized content (1MB limit is 1,048,576 bytes)
    doc_large = RawDocument(
        source="big.pdf",
        content="x" * (2 * 1024 * 1024),
        mime_type="application/pdf",
    )
    assert validator.validate(doc_large) is False

    # Unsupported MIME
    doc_bad_mime = RawDocument(
        source="john.pdf",
        content="hello",
        mime_type="image/png",
    )
    assert validator.validate(doc_bad_mime) is False

    # Unsupported Extension
    doc_bad_ext = RawDocument(
        source="john.txt",
        content="hello",
        mime_type="application/pdf",
    )
    assert validator.validate(doc_bad_ext) is False


def test_fixed_size_chunker():
    """Test character-based fixed size chunking."""
    config = ChunkingConfig(
        strategy=ChunkingStrategy.FIXED_SIZE,
        chunk_size=10,
        chunk_overlap=2,
        min_chunk_size=2,
    )
    chunker = FixedSizeChunker(config)
    text = "abcdefghijklmnop"
    chunks = chunker.chunk(text, document_id="doc1")
    
    assert len(chunks) > 1
    assert chunks[0].document_id == "doc1"
    assert chunks[0].chunk_index == 0
    # First chunk is "abcdefghij" (length 10)
    assert chunks[0].text == "abcdefghij"


def test_sentence_chunker():
    """Test regex sentence boundary chunking."""
    config = ChunkingConfig(
        strategy=ChunkingStrategy.SENTENCE,
        chunk_size=50,
        chunk_overlap=10,
        min_chunk_size=5,
    )
    chunker = SentenceChunker(config)
    text = "First sentence. Second sentence! Third sentence? Fourth."
    chunks = chunker.chunk(text, document_id="doc1")
    
    assert len(chunks) > 0
    assert any("First sentence." in c.text for c in chunks)
    assert all(c.metadata["chunk_strategy"] == "sentence" for c in chunks)


def test_paragraph_chunker():
    """Test paragraph chunking by double newlines."""
    # When chunk_size is large, paragraphs should merge
    config_merge = ChunkingConfig(
        strategy=ChunkingStrategy.PARAGRAPH,
        chunk_size=100,
        chunk_overlap=0,
        min_chunk_size=5,
    )
    chunker_merge = ParagraphChunker(config_merge)
    text = "Paragraph one text.\n\nParagraph two is here.\n\nThree."
    chunks_merge = chunker_merge.chunk(text, document_id="doc1")
    
    assert len(chunks_merge) == 1
    assert "Paragraph one text." in chunks_merge[0].text
    assert "Paragraph two is here." in chunks_merge[0].text

    # When chunk_size is small, paragraphs should split
    config_split = ChunkingConfig(
        strategy=ChunkingStrategy.PARAGRAPH,
        chunk_size=25,
        chunk_overlap=0,
        min_chunk_size=2,
    )
    chunker_split = ParagraphChunker(config_split)
    chunks_split = chunker_split.chunk(text, document_id="doc2")
    assert len(chunks_split) >= 2



def test_resume_section_chunker():
    """Test section detection on a mock resume."""
    config = ChunkingConfig(
        strategy=ChunkingStrategy.SECTION_AWARE,
        min_chunk_size=5,
        max_chunk_size=1000,
    )
    chunker = ResumeSectionChunker(config)
    resume = (
        "John Doe\nEmail: john@doe.com\n"
        "WORK EXPERIENCE\n"
        "Software Engineer at Acme Corp.\n"
        "Developed RAG pipelines in Python.\n"
        "EDUCATION\n"
        "BS in Computer Science from Stanford University."
    )
    chunks = chunker.chunk(resume, document_id="res1")
    
    # We expect:
    # 1. Preamble (John Doe...) -> chunk_section: header
    # 2. Experience -> chunk_section: work experience
    # 3. Education -> chunk_section: education
    sections = [c.metadata.get("chunk_section") for c in chunks]
    assert "header" in sections
    assert "work experience" in sections
    assert "education" in sections


def test_jd_section_chunker():
    """Test section detection on a mock Job Description."""
    config = ChunkingConfig(
        strategy=ChunkingStrategy.SECTION_AWARE,
        min_chunk_size=5,
        max_chunk_size=1000,
    )
    chunker = JDSectionChunker(config)
    jd = (
        "Senior Python Engineer Needed\n"
        "RESPONSIBILITIES\n"
        "- Build scalable REST APIs using FastAPI\n"
        "- Coordinate with frontend developers\n"
        "REQUIREMENTS\n"
        "- 5+ years of experience\n"
        "- Expert level in Python and SQL"
    )
    chunks = chunker.chunk(jd, document_id="jd1")
    
    sections = [c.metadata.get("chunk_section") for c in chunks]
    assert "responsibilities" in sections
    assert "requirements" in sections


def test_resume_metadata_extractor():
    """Test regex extraction of fields from a resume."""
    extractor = ResumeMetadataExtractor()
    text = (
        "Alice Smith\n"
        "alice.smith@example.com\n"
        "(123) 456-7890\n"
        "Python Backend Developer\n"
        "TECHNICAL SKILLS\n"
        "Python, FastAPI, SQL, Docker, Git, AWS\n"
        "EXPERIENCE\n"
        "Senior Developer with 7 years of experience in coding.\n"
        "EDUCATION\n"
        "Holds a Master's degree in engineering from MIT."
    )
    doc = RawDocument(source="alice_resume.docx", content=text)
    meta = extractor.extract(doc)
    
    assert meta["candidate_name"] == "Alice Smith"
    assert meta["email"] == "alice.smith@example.com"
    assert meta["phone"] == "(123) 456-7890"
    assert meta["experience_years"] == 7
    assert meta["education_level"] == "master"
    assert meta["job_title"] == "Python Backend Developer"
    # Skills parser should extract from the section
    assert "Python" in meta["skills"]
    assert "FastAPI" in meta["skills"]


def test_jd_metadata_extractor():
    """Test regex extraction of fields from a job description."""
    extractor = JDMetadataExtractor()
    text = (
        "Data Scientist\n"
        "About the company: We do AI.\n"
        "REQUIREMENTS\n"
        "• Python and PyTorch expertise\n"
        "• PhD in Statistics or Computer Science\n"
        "• 3+ years experience with machine learning\n"
    )
    doc = RawDocument(source="data_science_jd.pdf", content=text)
    meta = extractor.extract(doc)
    
    assert meta["job_title"] == "Data Scientist"
    assert meta["experience_years"] == 3
    assert meta["education_level"] == "phd"
    assert meta["industry"] == "technology"  # due to technology keywords
    assert any("PyTorch" in s for s in meta["required_skills"])


class MockLLMClient(LLMClient):
    """Mock LLM client implementation."""
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, prompt: str, *, system: str = "") -> str:
        return self.response


def test_llm_resume_metadata_extractor():
    """Test LLM-backed metadata extraction with mock response."""
    mock_response = (
        '{"candidate_name": "Bob Vance", "job_title": "Refrigeration Specialist", '
        '"skills": ["Sales", "Cooling"], "experience_years": 15, '
        '"education_level": "bachelor", "industry": "Refrigeration"}'
    )
    client = MockLLMClient(mock_response)
    extractor = LLMResumeMetadataExtractor(llm_client=client)
    
    text = "Bob Vance\nbob@vance.com\n15 years of experience\nBS degree."
    doc = RawDocument(source="bob.pdf", content=text)
    meta = extractor.extract(doc)
    
    assert meta["candidate_name"] == "Bob Vance"
    assert meta["job_title"] == "Refrigeration Specialist"
    assert meta["skills"] == ["Sales", "Cooling"]
    assert meta["experience_years"] == 15
    assert meta["education_level"] == "bachelor"
    assert meta["industry"] == "Refrigeration"


def test_llm_resume_metadata_extractor_fallback():
    """Test fallback to regex if LLM client throws exception."""
    class BadLLMClient(LLMClient):
        def complete(self, prompt: str, *, system: str = "") -> str:
            raise Exception("LLM server down")

    extractor = LLMResumeMetadataExtractor(llm_client=BadLLMClient())
    text = "Alice Smith\nalice.smith@example.com\n3+ years experience"
    doc = RawDocument(source="alice.pdf", content=text)
    meta = extractor.extract(doc)
    
    # Should use regex fallback
    assert meta["candidate_name"] == "Alice Smith"
    assert meta["experience_years"] == 3
    assert meta["email"] == "alice.smith@example.com"
