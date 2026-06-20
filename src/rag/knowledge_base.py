import json
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from collections import Counter
from enum import Enum

# -- Enums for Metadata & Priority --
class SourceTrustLevel(Enum):
    HIGH = 10    # WHO, CDC, NHS
    MED_HIGH = 8 # MedlinePlus, Mayo Clinic
    MED = 6      # General journals
    MED_LOW = 4  # WebMD, Healthline
    LOW = 2      # User-generated, synthetic

class EvidenceLevel(Enum):
    HIGH = "1a"
    MODERATE = "2a"
    LOW = "3a"
    NONE = "4a"

class ClinicalPriority(Enum):
    EMERGENCY = 10
    URGENT = 9
    ROUTINE = 8
    INFO = 6
    LOW = 4

# -- Medical Taxonomy Normalizer --
# Maps common lay terms to clinical terms for better retrieval
MEDICAL_TAXONOMY = {
    "heart attack": "myocardial infarction",
    "stroke": "cerebrovascular accident",
    "heartburn": "gastroesophageal reflux",
    "cold": "upper respiratory tract infection (viral)",
    "flu": "influenza",
    "back pain": "lumbar strain",
    "kidney stones": "nephrolithiasis",
    "allergic reaction": "anaphylaxis",
    "high blood pressure": "hypertension",
    "low blood sugar": "hypoglycemia"
}

def normalize_term(term: str) -> str:
    """Maps layman terms to clinical synonyms."""
    term_clean = term.lower().strip()
    if term_clean in MEDICAL_TAXONOMY:
        return MEDICAL_TAXONOMY[term_clean]
    return term_clean

@dataclass
class MedicalChunk:
    """
    Represents a single, valid, enriched chunk ready for embedding.
    """
    doc_id: str
    chunk_id: str
    text: str
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        return self.text

# -- Main Knowledge Base Class --
class MedicalKnowledgeBase:
    def __init__(self, output_dir: str = "./data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.documents: List[Dict[str, Any]] = []
        self.chunks: List[MedicalChunk] = []
        
        # Configuration
        self.max_chunk_size = 800
        self.chunk_overlap = 150

        # TRUST MAPPING

        self.source_trust = {
            "who": SourceTrustLevel.HIGH.value,
            "cdc": SourceTrustLevel.HIGH.value,
            "nhs": SourceTrustLevel.MED_HIGH.value,
            "mayo clinic": SourceTrustLevel.MED_HIGH.value,
            "medlineplus": SourceTrustLevel.MED_HIGH.value,
            "webmd": SourceTrustLevel.MED_LOW.value,
            "healthline": SourceTrustLevel.MED_LOW.value,
        }

       
        # DOMAIN PRIORITIES
        self.domain_priorities = {
            "cardiovascular": ClinicalPriority.EMERGENCY.value,
            "infectious_disease": ClinicalPriority.URGENT.value,
            "diabetes": ClinicalPriority.ROUTINE.value,
            "general_medicine": ClinicalPriority.ROUTINE.value,
            "mental_health": ClinicalPriority.ROUTINE.value,
            "global_health": ClinicalPriority.ROUTINE.value,
            "pharmacology": ClinicalPriority.ROUTINE.value,
        }
        
        # Dangerous content disclaimers
        self.dangerous_disclaimers = {
            "emergency": "⚠️ EMERGENCY CONTENT: This information may indicate a life-threatening situation. Always call 911 or seek immediate medical attention if you suspect an emergency.",
            "high": "⚠️ HIGH-RISK CONTENT: This information may indicate a serious condition. If you are experiencing severe symptoms, please seek urgent medical care.",
            "general": "⚠️ GENERAL MEDICAL INFORMATION: This content is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."
        }

    def detect_source(self, file_path: Path, content: str) -> str:
        """
        Reliable source detection.
        """

        filename = file_path.stem.lower()
        first_section = content[:500].lower()

        if (
            "who" in filename
            or "world health organization" in first_section
        ):
            return "WHO"

        if (
            "cdc" in filename
            or "centers for disease control" in first_section
        ):
            return "CDC"

        if (
            "nhs" in filename
            or "national health service" in first_section
        ):
            return "NHS"

        if "mayo clinic" in filename or "mayo" in filename:
            return "Mayo Clinic"

        if "medlineplus" in filename:
            return "MedlinePlus"

        if "webmd" in filename:
            return "WebMD"

        if "healthline" in filename:
            return "Healthline"

        return "Unknown"


    def determine_domain(self, text: str) -> str:

        text = text.lower()

        domain_keywords = {
            "cardiovascular": [
                "cpr",
                "heart attack",
                "stroke",
                "chest pain",
                "aed",
                "cardiac arrest",
            ],
            "infectious_disease": [
                "infection",
                "virus",
                "bacteria",
                "malaria",
                "hepatitis",
                "vaccination",
            ],
            "diabetes": [
                "diabetes",
                "insulin",
                "glucose",
                "hyperglycemia",
                "hypoglycemia",
            ],
            "mental_health": [
                "depression",
                "anxiety",
                "therapy",
                "psychosis",
            ],
            "pharmacology": [
                "drug",
                "medication",
                "dosage",
                "prescription",
            ],
            "global_health": [
                "pandemic",
                "epidemic",
                "public health",
                "outbreak",
            ],
            "general_medicine": [
                "first aid",
                "burn",
                "injury",
                "wound",
                "fever",
            ],
        }

        scores = {}

        for domain, keywords in domain_keywords.items():
            scores[domain] = sum(
                keyword in text
                for keyword in keywords
            )

        best_domain = max(scores, key=scores.get)

        if scores[best_domain] == 0:
            return "general_medicine"

        return best_domain


    def detect_risk_level(self, text: str) -> str:

        text = text.lower()

        emergency_keywords = [
            "call 911",
            "call emergency services",
            "cardiac arrest",
            "cpr",
            "unconscious",
            "not breathing",
        ]

        high_keywords = [
            "stroke",
            "heart attack",
            "chest pain",
            "severe bleeding",
            "anaphylaxis",
        ]

        if any(k in text for k in emergency_keywords):
            return "emergency"

        if any(k in text for k in high_keywords):
            return "high"

        return "low"


    def normalize_medical_terms(self, content: str) -> str:

        for lay_term, clinical_term in MEDICAL_TAXONOMY.items():

            content = re.sub(
                rf"\b{re.escape(lay_term)}\b",
                clinical_term,
                content,
                flags=re.IGNORECASE,
            )

        return content


    def build_metadata(
        self,
        file_path: Path,
        content: str,
    ):

        source_name = self.detect_source(
            file_path,
            content,
        )

        source_key = source_name.lower()

        trust_level = self.source_trust.get(
            source_key,
            SourceTrustLevel.MED_LOW.value,
        )

        is_trusted = (
            trust_level
            >= SourceTrustLevel.MED_HIGH.value
        )

        domain = self.determine_domain(content)

        risk_level = self.detect_risk_level(content)

        priority = self.domain_priorities.get(
            domain,
            ClinicalPriority.ROUTINE.value,
        )

        # Risk overrides

        if risk_level == "emergency":
            priority = ClinicalPriority.EMERGENCY.value

        elif risk_level == "high":
            priority = max(
                priority,
                ClinicalPriority.URGENT.value,
            )

        return {
            "source": source_name,
            "domain": domain,
            "trust_level": trust_level,
            "priority": priority,
            "risk_level": risk_level,
            "is_trusted_source": is_trusted,
        }

    # -- Loading & Validation --
    def load_and_enrich_document(
        self,
        file_path: str,
    ) -> None:
        """
        Load document (TXT, MD, or PDF), enrich metadata,
        normalize medical terminology, and prepare for chunking.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"⚠️ File not found: {file_path}")
            return

        # --------------------------------------------------
        # Seamless PDF and Plain Text Extension Handling
        # --------------------------------------------------
        content = ""
        if file_path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                
                # Extract characters from each page sequentially
                text_pages = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_pages.append(page_text)
                        
                content = "\n\n".join(text_pages)
                print(f"📄 Successfully extracted {len(reader.pages)} pages from PDF: {file_path.name}")
            except Exception as e:
                print(f"⚠️ Failed to parse PDF structure for {file_path.name}: {e}")
                return
        else:
            try:
                if file_path.suffix.lower() == ".md":

                    import frontmatter

                    post = frontmatter.load(file_path)

                    content = post.content

                    markdown_metadata = dict(post.metadata)

                else:
                    content = file_path.read_text(
                        encoding="utf-8"
                    )

                    markdown_metadata = {}

            except Exception as e:
                print(
                    f"⚠️ Failed to read text file "
                    f"{file_path.name}: {e}"
                )
                return

        # Verify that text extraction actually yielded readable content
        if not content.strip():
            print(f"⚠️ Skipping execution for {file_path.name}: No valid textual content extracted.")
            return

        # --------------------------------------------------
        # Title Extraction Logic
        # --------------------------------------------------
        title = file_path.stem

        for line in content.splitlines():
            line = line.strip()
            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
                break
            if line.startswith("#"):
                title = line.replace("#", "").strip()
                break

        # --------------------------------------------------
        # Metadata Construction & Taxonomy Alignment
        # --------------------------------------------------
        if file_path.suffix.lower() == ".md":

            metadata = {
                **markdown_metadata,

                "domain": file_path.parent.name,

                "trust_level": markdown_metadata.get(
                    "trust_score",
                    SourceTrustLevel.MED_HIGH.value
                ),

                "risk_level": self.detect_risk_level(
                    content
                ),

                "priority": self.domain_priorities.get(
                    file_path.parent.name,
                    ClinicalPriority.ROUTINE.value
                ),
            }

            metadata["is_trusted_source"] = (
                metadata["trust_level"]
                >= SourceTrustLevel.MED_HIGH.value
            )

        else:

            metadata = self.build_metadata(
                file_path=file_path,
                content=content,
            )
        normalized_content = (
            self.normalize_medical_terms(content)
        )

        # --------------------------------------------------
        # Assemble Document Data Object
        # --------------------------------------------------
        doc_data = {
            "id": str(uuid.uuid4()),
            "title": title,
            "content": content,
            "metadata": {
                **metadata,
                "normalized_content": normalized_content,
                "disclaimer_tag": metadata["risk_level"],
            "file_type": file_path.suffix.lower()
            },
        }

        self.documents.append(doc_data)

        print(f"✅ Loaded & Enriched: {title}")
        print(f"   Source    : {metadata['source']}")
        print(f"   Domain    : {metadata['domain']}")
        print(f"   Trust     : {metadata['trust_level']}")
        print(f"   Priority  : {metadata['priority']}")
        print(f"   Risk      : {metadata['risk_level']}\n")

    def load_documents(self, file_paths: List[str]) -> None:
        """Load multiple files."""
        for fp in file_paths:
            self.load_and_enrich_document(fp)
        print(f"✅ Loaded {len(self.documents)} documents with enriched metadata.")

    # -- Section-Aware Chunking --
    def generate_chunks(self) -> List[MedicalChunk]:
        """Generates chunks with section-aware splitting for Markdown and intelligent splitting for other formats. Each chunk is enriched with comprehensive metadata for RAG optimization."""
        
        
        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter,
            MarkdownHeaderTextSplitter
        )

        generic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

        self.chunks = []

        for doc in self.documents:

            content = doc["metadata"]["normalized_content"]

            doc_id = doc["id"]

            priority = doc["metadata"].get(
                "priority",
                ClinicalPriority.ROUTINE.value
            )

            trust = doc["metadata"].get(
                "trust_level",
                2
            )

            risk_level = doc["metadata"].get(
                "risk_level",
                "low"
            )

            file_type = doc.get("file_type", "")

            # --------------------------------------------------
            # Markdown files
            # --------------------------------------------------
            if file_type == ".md":

                md_splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=[
                        ("#", "title"),
                        ("##", "section"),
                    ]
                )

                md_chunks = md_splitter.split_text(content)

                for i, chunk in enumerate(md_chunks):

                    chunk_text = chunk.page_content.strip()

                    if not chunk_text:
                        continue

                    section = chunk.metadata.get(
                        "section",
                        "general"
                    )

                    chunk_obj = MedicalChunk(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}_{i}",
                        text=chunk_text,
                        metadata={
                            "source": doc["metadata"]["source"],
                            "domain": doc["metadata"]["domain"],
                            "trust_level": trust,
                            "priority": priority,
                            "doc_title": doc["title"],
                            "chunk_index": i,
                            "risk_level": risk_level,
                            "disclaimer_tag": risk_level,

                            # NEW
                            "section": section,

                            # YAML metadata
                            "condition": doc["metadata"].get(
                                "condition"
                            ),
                            "content_type": doc["metadata"].get(
                                "content_type"
                            ),
                            "last_reviewed": doc["metadata"].get(
                                "last_reviewed"
                            ),
                        }
                    )

                    self.chunks.append(chunk_obj)

            # --------------------------------------------------
            # TXT / PDF / future formats
            # --------------------------------------------------
            else:

                raw_chunks = generic_splitter.split_text(
                    content
                )

                for i, chunk_text in enumerate(raw_chunks):

                    if not chunk_text.strip():
                        continue

                    chunk_obj = MedicalChunk(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}_{i}",
                        text=chunk_text.strip(),
                        metadata={
                            "source": doc["metadata"]["source"],
                            "domain": doc["metadata"]["domain"],
                            "trust_level": trust,
                            "priority": priority,
                            "doc_title": doc["title"],
                            "chunk_index": i,
                            "risk_level": risk_level,
                            "disclaimer_tag": risk_level,
                        }
                    )

                    self.chunks.append(chunk_obj)

        return self.chunks

    # -- Safety Layer (Runtime) --
    def get_runtime_disclaimer(self, chunk_metadata: Dict) -> str:
        """
        Returns the appropriate disclaimer string based on the metadata risk level.
        This is called AFTER retrieval, before sending to the user.
        """
        risk = chunk_metadata.get("disclaimer_tag", chunk_metadata.get("risk_level", "low"))
        
        # Normalize 'emergency' tag
        if risk == "emergency":
            return self.dangerous_disclaimers["emergency"] + "\n\n"
        elif risk == "high":
            return self.dangerous_disclaimers["high"] + "\n\n"
        else:
            return self.dangerous_disclaimers["general"] + "\n\n"

    def apply_safety_layer(self, retrieved_chunk_text: str, chunk_metadata: Dict) -> str:
        """
        Combines retrieved text with the runtime disclaimer.
        Disclaimer is now PREPENDED for safety.
        """
        disclaimer = self.get_runtime_disclaimer(chunk_metadata)
        return f"{disclaimer}{retrieved_chunk_text}"

    # -- Export --
    def save_chunks_for_rag(self, output_file: str = "chunks_rag.json") -> None:
        """
        Saves chunks optimized for RAG (JSON format).
        Includes priority and trust level for reranking.
        """
        file_path = self.output_dir / output_file
        data = []
        
        for chunk in self.chunks:
            # asdict handles the dataclass -> dict conversion correctly now
            entry = asdict(chunk)
            data.append(entry)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"📄 Saved {len(data)} optimized chunks to {file_path}")
        
        # FIX: Calculate stats correctly on the list of dicts
        if data:
            priorities = [c.get("metadata", {}).get("priority", 6) for c in data]
            domains = [c.get("metadata", {}).get("domain", "unknown") for c in data]
            trust_levels = [c.get("metadata", {}).get("trust_level", 2) for c in data]
            
            print(f"   📊 Avg Priority: {sum(priorities) / len(priorities):.2f}")
            print(f"   📋 Domains: {dict(Counter(domains))}")
            print(f"   🛡️ Trust Levels: {dict(Counter(trust_levels))}")

    def print_statistics(self) -> None:
        """Prints a summary of the ingestion."""
        if not self.chunks:
            print("⚠️ No chunks generated.")
            return

        priorities = [c.metadata.get("priority", 6) for c in self.chunks]
        domains = [c.metadata.get("domain", "unknown") for c in self.chunks]
        trust_levels = [c.metadata.get("trust_level", 2) for c in self.chunks]
        
        print("\n" + "="*60)
        print("🏥 MEDICAL RAG PIPELINE STATISTICS")
        print("="*60)
        print(f"Total Chunks: {len(self.chunks)}")
        print(f"Unique Domains: {dict(Counter(domains))}")
        print(f"Source Trust Distribution: {dict(Counter(trust_levels))}")
        print(f"Priority Distribution: {dict(Counter(priorities))}")
        print("="*60 + "\n")