# src/tools/web_scraper.py
import logging
import re
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from ddgs import DDGS
import requests

from config.settings import TRUSTED_WEB_SOURCES

logger = logging.getLogger("medigemma_scraper")

class MedicalWebScraper:
    """
    Scrapes safe, whitelisted institutional clinical and nutrition data live
    as a backup fallback whenever local database lookups fail.
    """

    # Domain groupings for domain-specific search routing
    NUTRITION_DOMAINS = ["healthline.com", "hsph.harvard.edu", "eatright.org", "webmd.com"]
    CLINICAL_DOMAINS = ["nhs.uk", "who.int", "mayoclinic.org"]

    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }

    def fetch_whitelisted_urls(self, query: str, max_results: int = 2, domain_type: str = "clinical") -> List[str]:
        """
        Executes a SINGLE aggregated query to DDGS restricted to specific whitelisted sites.
        Prevents rate-limiting (429) and correctly respects query domain intent.
        """
        # Select target domains based on intent
        allowed_domains = self.NUTRITION_DOMAINS if domain_type == "nutrition" else self.CLINICAL_DOMAINS
        
        # Build unified search filter: e.g. "eggs nutrients (site:healthline.com OR site:hsph.harvard.edu)"
        site_filter = " OR ".join([f"site:{d}" for d in allowed_domains])
        full_query = f"{query} ({site_filter})"
        
        logger.info(f"🌐 Dispatching unified DDGS query [{domain_type.upper()}]: [ {full_query} ]")
        
        urls = []
        try:
            with DDGS() as ddgs:
                # Single search pass fetching top candidates
                search_results = list(ddgs.text(full_query, max_results=max_results * 2))
                
                if not search_results:
                    logger.warning(f"⚠️ DDGS extracted zero results for: {full_query}")
                    return []
                
                for result in search_results:
                    href = result.get("href", "")
                    parsed = urlparse(href)
                    domain = parsed.netloc.replace("www.", "")
                    
                    # Verify link belongs to an allowed domain and is valid HTML
                    if any(target in domain for target in allowed_domains) and not href.endswith((".pdf", ".xml", ".jpg", ".png")):
                        if href not in urls and href.startswith("http"):
                            urls.append(href)
                            if len(urls) >= max_results:
                                break
        except Exception as e:
            logger.error(f"❌ Failed during DDGS extraction pass: {e}")

        # Deduplicate and return final list
        urls = list(dict.fromkeys(urls))
        logger.info(f"🎯 Whitelisted target URLs finalized: {urls}")
        return urls

    def scrape_to_structured_markdown(self, url: str, query_condition: str) -> List[Dict[str, Any]]:
        """
        Scrapes raw web pages, extracts clean prose, and builds structured,
        metadata-enriched chunks for RAG embedding.
        """
        logger.info(f"📄 Re-structuring text layers into unified MD format from: {url}")
        try:
            response = self.session.get(url, headers=self.headers, timeout=6)
            if response.status_code != 200:
                return []
                
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract title
            page_title = soup.find("h1")
            title_text = page_title.get_text(strip=True) if page_title else query_condition.title()
            
            # Strip non-semantic elements
            for tag in soup(["script", "style", "nav", "header", "footer", "form", "aside", "iframe"]):
                tag.extract()
                
            # Collect meaningful paragraphs (> 40 chars)
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
            if not paragraphs:
                return []

            # Dynamic domain metadata lookup via central settings
            parsed_domain = urlparse(url).netloc.replace("www.", "")
            matched_key = next((d for d in TRUSTED_WEB_SOURCES if d in parsed_domain), "Trusted Source")
            source_info = TRUSTED_WEB_SOURCES.get(matched_key, {"name": "Trusted Source", "score": 7})
            
            source_org = source_info["name"]
            trust_score = source_info["score"]
            current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            # Combine paragraphs clean into bullet points
            body_content = "\n".join([f"- {p}" for p in paragraphs])

            # Markdown structure
            structured_doc = (
                f"# {title_text}\n\n"
                f"## Live Reference Summary ({source_org})\n"
                f"{body_content}"
            )
            
            # Paragraph-aware chunking (preserves complete sentences)
            chunks_payload = []
            chunk_units = []
            current_len = 0

            for p in paragraphs:
                if current_len + len(p) > 600 and chunk_units:
                    chunk_body = "\n".join(chunk_units)
                    semantic_text = f"Context: {query_condition.upper()} guidance from {source_org}.\n{chunk_body}"
                    
                    # Deterministic sha256 checksum ID
                    chunk_id = hashlib.sha256(semantic_text.encode("utf-8")).hexdigest()
                    
                    chunks_payload.append({
                        "id": chunk_id,
                        "text": semantic_text,
                        "metadata": {
                            "source": source_org,
                            "url": url,
                            "trust_score": trust_score,
                            "content_type": "cached_web_fallback",
                            "condition": query_condition.lower()
                        }
                    })
                    chunk_units = []
                    current_len = 0
                
                chunk_units.append(f"- {p}")
                current_len += len(p)

            # Flush residual chunk
            if chunk_units:
                chunk_body = "\n".join(chunk_units)
                semantic_text = f"Context: {query_condition.upper()} guidance from {source_org}.\n{chunk_body}"
                chunk_id = hashlib.sha256(semantic_text.encode("utf-8")).hexdigest()
                chunks_payload.append({
                    "id": chunk_id,
                    "text": semantic_text,
                    "metadata": {
                        "source": source_org,
                        "url": url,
                        "trust_score": trust_score,
                        "content_type": "cached_web_fallback",
                        "condition": query_condition.lower()
                    }
                })

            return chunks_payload
            
        except Exception as e:
            logger.error(f"❌ Structured markdown parsing failed for {url}: {e}")
            return []