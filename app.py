"""
Academic Paper Mass Downloader
--------------------------------
Searches multiple open-access scholarly APIs (OpenAlex, Crossref, Unpaywall,
CORE, DOAJ, OpenAIRE) for papers/theses matching user criteria, deduplicates
results, downloads legally-available open-access PDFs (with retries and
multi-source fallback), and merges them into a single PDF (with a generated
table of contents) plus a metadata/bibliography TXT and CSV export.

Run with: streamlit run app.py
"""

import io
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfWriter, PdfReader
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT
from xml.sax.saxutils import escape as _xml_escape

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

APP_TITLE = "Open-Access Paper Mass Downloader"
USER_AGENT = "PaperMassDownloader/1.0 (mailto:research-tool@example.com)"
DEFAULT_TIMEOUT = 20
HEADERS = {"User-Agent": USER_AGENT}
MAX_RETRIES_PER_URL = 3
RETRY_BACKOFF_SECONDS = 1.5

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Paper:
    title: str
    authors: list = field(default_factory=list)
    year: str = ""
    venue: str = ""
    doi: str = ""
    country: str = ""
    institution: str = ""
    subject_field: str = ""
    source: str = ""
    oa_urls: list = field(default_factory=list)   # ordered candidate PDF URLs, tried in sequence
    landing_url: str = ""
    abstract: str = ""
    doc_type: str = ""      # "research" | "thesis" | "conference" | "report"
    is_oa: bool = True      # whether an open-access copy is believed to exist at all
    pages: str = ""

    def key(self):
        if self.doi:
            return "doi:" + self.doi.lower().strip()
        norm = unicodedata.normalize("NFKD", self.title or "").lower()
        norm = re.sub(r"[^a-z0-9]+", "", norm)
        return "title:" + norm[:120]

    def citation_line(self):
        auth = ", ".join(self.authors[:3]) + (" et al." if len(self.authors) > 3 else "")
        bits = [b for b in [self.title, auth, self.venue, self.year] if b]
        return " — ".join(bits)

    def bibliography_entry(self, index):
        auth = ", ".join(self.authors) if self.authors else "Unknown author"
        parts = [f"[{index}] {auth} ({self.year or 'n.d.'}). {self.title}."]
        if self.venue:
            parts.append(f" {self.venue}.")
        if self.doi:
            parts.append(f" https://doi.org/{self.doi}")
        return "".join(parts)

    def add_candidate_url(self, url):
        if url and url not in self.oa_urls:
            self.oa_urls.append(url)


# --------------------------------------------------------------------------
# Query expansion (User query -> search -> extract terms -> suggest expanded queries)
# --------------------------------------------------------------------------

STOPWORDS = set("""
a an the of and or for to in on with from by is are was were be been being
this that these those as at into using use study analysis review paper
research thesis effect effects based approach method methods results
""".split())


def extract_keywords(papers, top_n=12):
    """Cheap term-frequency extraction over candidate titles to suggest related terms."""
    counts = {}
    for p in papers:
        words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", p.title or "")
        seen_in_title = set()
        for w in words:
            wl = w.lower()
            if wl in STOPWORDS or len(wl) < 4:
                continue
            if wl in seen_in_title:
                continue
            seen_in_title.add(wl)
            counts[wl] = counts.get(wl, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w, c in ranked if c >= 2][:top_n]


def suggest_expanded_queries(original_keyword, extracted_terms):
    suggestions = []
    for term in extracted_terms[:6]:
        if term.lower() in original_keyword.lower():
            continue
        suggestions.append(f"{original_keyword} {term}")
    return suggestions


# --------------------------------------------------------------------------
# Source clients — each returns a list[Paper]
# --------------------------------------------------------------------------

DOC_TYPE_LABELS = {
    "research": "Research papers",
    "thesis": "Theses / dissertations",
    "conference": "Conference papers",
    "report": "Reports",
}


def search_openalex(keyword, year_from, year_to, country, institution, field_name, limit, doc_type="all"):
    papers = []
    try:
        filters = [f"from_publication_date:{year_from}-01-01",
                   f"to_publication_date:{year_to}-12-31"]
        if country:
            filters.append(f"institutions.country_code:{country}")
        # Real OpenAlex `type` vocabulary (Crossref-based): article, dissertation,
        # proceedings-article, report, book, etc. "journal-article" and "paratext"
        # are not valid values here and were silently producing zero matches.
        type_map = {"thesis": "dissertation", "research": "article",
                    "conference": "proceedings-article", "report": "report"}
        if doc_type != "all" and doc_type in type_map:
            filters.append(f"type:{type_map[doc_type]}")
        params = {
            "search": keyword,
            "filter": ",".join(filters),
            "per-page": min(limit, 100),
            "mailto": "research-tool@example.com",
        }
        r = requests.get("https://api.openalex.org/works", params=params,
                          headers=HEADERS, timeout=DEFAULT_TIMEOUT)
        if r.status_code != 200:
            st.session_state.setdefault("errors", []).append(
                f"OpenAlex: HTTP {r.status_code} — {r.text[:200]}")
            return papers
        for w in r.json().get("results", []):
            title = w.get("title") or ""
            if institution:
                inst_match = any(
                    institution.lower() in (a.get("institutions", [{}])[0].get("display_name", "") or "").lower()
                    for a in w.get("authorships", []) if a.get("institutions")
                )
                if not inst_match:
                    continue
            authors = [a.get("author", {}).get("display_name", "") for a in w.get("authorships", [])]
            oa = w.get("open_access", {}) or {}
            best_loc = w.get("best_oa_location") or {}
            biblio = w.get("biblio", {}) or {}
            first_page, last_page = biblio.get("first_page"), biblio.get("last_page")
            pages = f"{first_page}-{last_page}" if first_page and last_page else ""
            raw_type = (w.get("type") or "")
            if raw_type == "dissertation":
                ptype = "thesis"
            elif raw_type in ("proceedings-article", "paratext"):
                ptype = "conference"
            elif raw_type == "report":
                ptype = "report"
            else:
                ptype = "research"
            p = Paper(
                title=title,
                authors=[a for a in authors if a],
                year=str(w.get("publication_year") or ""),
                venue=(w.get("host_venue", {}) or {}).get("display_name", "") or
                      (w.get("primary_location", {}) or {}).get("source", {}).get("display_name", "") or "",
                doi=(w.get("doi") or "").replace("https://doi.org/", ""),
                country=country,
                institution=institution,
                subject_field=field_name,
                source="OpenAlex",
                landing_url=w.get("id", ""),
                is_oa=bool(oa.get("is_oa")),
                pages=pages,
                doc_type=ptype,
            )
            p.add_candidate_url(best_loc.get("pdf_url"))
            p.add_candidate_url(oa.get("oa_url"))
            papers.append(p)
    except Exception as e:
        st.session_state.setdefault("errors", []).append(f"OpenAlex: {e}")
    return papers


def search_crossref(keyword, year_from, year_to, limit, doc_type="all"):
    papers = []
    try:
        filt = f"from-pub-date:{year_from}-01-01,until-pub-date:{year_to}-12-31"
        type_map = {"thesis": "dissertation", "research": "journal-article",
                    "conference": "proceedings-article", "report": "report"}
        if doc_type != "all" and doc_type in type_map:
            filt += f",type:{type_map[doc_type]}"
        params = {"query": keyword, "filter": filt, "rows": min(limit, 100)}
        r = requests.get("https://api.crossref.org/works", params=params,
                          headers=HEADERS, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        for item in r.json().get("message", {}).get("items", []):
            title = (item.get("title") or [""])[0]
            authors = [f"{a.get('given','')} {a.get('family','')}".strip()
                       for a in item.get("author", [])] if item.get("author") else []
            link_pdf = ""
            for link in item.get("link", []) or []:
                if "pdf" in (link.get("content-type", "") or "").lower():
                    link_pdf = link.get("URL", "")
                    break
            raw_type = item.get("type") or ""
            if "dissertation" in raw_type:
                ptype = "thesis"
            elif "proceedings" in raw_type:
                ptype = "conference"
            elif "report" in raw_type:
                ptype = "report"
            else:
                ptype = "research"
            p = Paper(
                title=title,
                authors=authors,
                year=str((item.get("issued", {}).get("date-parts", [[None]])[0] or [""])[0] or ""),
                venue=(item.get("container-title") or [""])[0],
                doi=item.get("DOI", ""),
                source="Crossref",
                landing_url=item.get("URL", ""),
                pages=item.get("page", "") or "",
                doc_type=ptype,
                is_oa=bool(link_pdf),
            )
            p.add_candidate_url(link_pdf)
            papers.append(p)
    except Exception as e:
        st.session_state.setdefault("errors", []).append(f"Crossref: {e}")
    return papers


# --- CORE query-language helpers -------------------------------------------------
#
# FIX: The previous implementation crammed keyword + year-range + document-type
# into a single Elasticsearch-style CORE query string
# (e.g. "(enhanced oil recovery) AND yearPublished>=2015 AND yearPublished<=2026
# AND documentType:\"research\""). CORE's query parser is finicky about how
# free-text clauses combine with AND'd range/field filters, and this combined
# form was silently producing zero hits even with a valid key.
#
# The fix below follows the simpler, more reliable approach: send CORE a plain
# free-text query (no year/type filters baked into the query string at all),
# then apply year and document-type filtering locally in Python on the
# results that come back. This isolates whether a "no results" outcome is a
# genuine lack of matches vs. a query-syntax problem, and a debug panel is
# added below so HTTP status / raw response can be inspected directly.
#
# NOTE: unlike a stricter alternative that force-quotes every multi-word
# keyword as an exact phrase, we keep multi-word keywords unquoted so CORE's
# own relevance-ranked free-text search can match papers where the words
# appear near each other but not as a rigid literal phrase. This avoids
# swapping "zero results from bad query structure" for "zero results because
# an exact-phrase match is too strict."

CORE_DOC_TYPE_MAP = {
    "thesis": "thesis",
    "research": "research",
    "conference": "conference proceedings",
    "report": "report",
}

# Characters that are meaningful in CORE's Elasticsearch-style query syntax
# and need escaping when they appear inside plain user-typed keywords.
_CORE_SPECIAL_CHARS_RE = re.compile(r'([+\-!(){}\[\]^"~*?:\\/])')


def _build_core_query(keyword):
    """Build a conservative, free-text-only CORE query.

    Year and document-type filtering are intentionally NOT included here —
    they're applied locally after CORE returns results (see search_core).
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return "*"
    # Escape CORE/Elasticsearch special characters so stray punctuation in a
    # user's keyword doesn't break the query syntax.
    escaped = _CORE_SPECIAL_CHARS_RE.sub(r"\\\1", keyword)
    return escaped


def search_core(keyword, year_from, year_to, limit, api_key=None, doc_type="all"):
    """CORE's v3 API works without a key (100 tokens/day, no full text), but a
    free API key raises the daily token allowance considerably and is required
    to get full-text access. See https://core.ac.uk/services/api.

    Filtering strategy: send a broad free-text query to CORE, then filter the
    returned records locally by year and document type. This avoids relying
    on CORE's AND-combination of free-text + range + field filters in a single
    query string, which was the likely cause of spurious zero-result queries.
    """
    papers = []
    try:
        headers = dict(HEADERS)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        q = _build_core_query(keyword)
        # Fetch generously above `limit` since local year/type filtering will
        # discard some fraction of what CORE returns.
        fetch_limit = min(max(limit * 2, limit, 1), 100)
        params = {"q": q, "limit": fetch_limit}

        r = requests.get("https://api.core.ac.uk/v3/search/works", params=params,
                          headers=headers, timeout=DEFAULT_TIMEOUT)

        # Keep a debug record regardless of outcome so the "CORE debug" panel
        # in the UI can show exactly what happened on the most recent search.
        debug_entry = {
            "query": q,
            "status": r.status_code,
            "url": r.url,
            "response_preview": r.text[:1000],
        }
        st.session_state.setdefault("core_debug", []).append(debug_entry)

        if r.status_code == 401:
            st.session_state.setdefault("errors", []).append(
                "CORE: HTTP 401 — API key was rejected. Double-check the key is valid "
                "and copied without extra spaces.")
            return papers
        if r.status_code == 403:
            st.session_state.setdefault("errors", []).append(
                "CORE: HTTP 403 — access denied for this key/account.")
            return papers
        if r.status_code == 429:
            retry_after = r.headers.get("X-RateLimit-Retry-After", "unknown")
            st.session_state.setdefault("errors", []).append(
                f"CORE: rate-limited (429) — token allowance exhausted, retry after {retry_after}. "
                "Unauthenticated/free-tier keys get a limited number of tokens per day; "
                "see https://core.ac.uk/services/api for higher tiers.")
            return papers
        if r.status_code != 200:
            st.session_state.setdefault("errors", []).append(
                f"CORE: HTTP {r.status_code} — {r.text[:200]}")
            return papers

        results = r.json().get("results", [])
        if not results:
            st.session_state.setdefault("errors", []).append(
                f"CORE: query {q!r} returned 0 results from CORE itself (not a local "
                "filtering issue) — try a broader/shorter keyword.")
            return papers

        kept = 0
        for w in results:
            # --- local year filtering -----------------------------------
            year_raw = w.get("yearPublished") or w.get("year_published") or ""
            try:
                year_int = int(year_raw)
            except (TypeError, ValueError):
                year_int = None
            if year_int is not None and (year_int < year_from or year_int > year_to):
                continue

            # --- local document-type filtering --------------------------
            raw_type = w.get("documentType") or w.get("document_type") or ""
            if isinstance(raw_type, list):
                raw_type = raw_type[0] if raw_type else ""
            raw_type = str(raw_type).lower()
            if "thesis" in raw_type or "dissertation" in raw_type:
                ptype = "thesis"
            elif "conference" in raw_type:
                ptype = "conference"
            elif "report" in raw_type:
                ptype = "report"
            else:
                ptype = "research"
            if doc_type != "all" and ptype != doc_type:
                continue

            authors = w.get("authors") or []
            author_names = [a.get("name", "") if isinstance(a, dict) else str(a) for a in authors]

            source_urls = w.get("sourceFulltextUrls") or w.get("source_fulltext_urls") or []
            if isinstance(source_urls, str):
                source_urls = [source_urls]
            download_url = w.get("downloadUrl") or w.get("download_url") or ""

            p = Paper(
                title=w.get("title", "") or "",
                authors=author_names,
                year=str(year_raw),
                venue=w.get("publisher", "") or "",
                doi=w.get("doi", "") or "",
                source="CORE",
                landing_url=source_urls[0] if source_urls else download_url,
                doc_type=ptype,
                is_oa=True,
            )
            if download_url:
                p.add_candidate_url(download_url)
            for alt in source_urls:
                p.add_candidate_url(alt)
            papers.append(p)
            kept += 1
            if kept >= limit:
                break

        if results and kept == 0:
            st.session_state.setdefault("errors", []).append(
                f"CORE: found {len(results)} raw result(s) for {q!r}, but all were filtered "
                "out locally by year range / document type — try widening the year range "
                "or setting Document type to 'All'.")
    except Exception as e:
        st.session_state.setdefault("errors", []).append(f"CORE: {type(e).__name__}: {e}")
    return papers


def search_doaj(keyword, year_from, year_to, limit, doc_type="all"):
    """DOAJ indexes open-access journals only. Skipped unless 'research' or 'all' requested."""
    papers = []
    if doc_type not in ("all", "research"):
        return papers
    try:
        r = requests.get(f"https://doaj.org/api/search/articles/{requests.utils.quote(keyword)}",
                          params={"pageSize": min(limit, 100)}, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        for res in r.json().get("results", []):
            bib = res.get("bibjson", {})
            year = bib.get("year", "")
            if year and (int(year) < year_from or int(year) > year_to):
                continue
            pdf_url = ""
            for link in bib.get("link", []) or []:
                if link.get("type") == "fulltext":
                    pdf_url = link.get("url", "")
                    break
            p = Paper(
                title=bib.get("title", ""),
                authors=[a.get("name", "") for a in (bib.get("author") or [])],
                year=str(year),
                venue=(bib.get("journal", {}) or {}).get("title", ""),
                doi=next((i.get("id", "") for i in bib.get("identifier", []) if i.get("type") == "doi"), ""),
                source="DOAJ",
                landing_url=pdf_url,
                doc_type="research",
                pages=f"{bib.get('start_page','')}-{bib.get('end_page','')}".strip("-"),
                is_oa=True,
            )
            p.add_candidate_url(pdf_url)
            papers.append(p)
    except Exception as e:
        st.session_state.setdefault("errors", []).append(f"DOAJ: {e}")
    return papers


def search_openaire(keyword, year_from, year_to, limit, doc_type="all"):
    papers = []
    try:
        params = {
            "keywords": keyword,
            "fromDateAccepted": f"{year_from}-01-01",
            "toDateAccepted": f"{year_to}-12-31",
            "size": min(limit, 100),
            "format": "json",
        }
        r = requests.get("https://api.openaire.eu/search/publications", params=params,
                          headers=HEADERS, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        results = (data.get("response", {}).get("results", {}) or {}).get("result", [])
        for res in results:
            meta = res.get("metadata", {}).get("oaf:entity", {}).get("oaf:result", {})
            title = ""
            titles = meta.get("title", "")
            if isinstance(titles, list):
                title = titles[0].get("$", "") if titles and isinstance(titles[0], dict) else ""
            elif isinstance(titles, dict):
                title = titles.get("$", "")
            if not title:
                continue
            candidate_urls = []
            children = meta.get("children", {})
            instances = children.get("instance", []) if isinstance(children, dict) else []
            if isinstance(instances, dict):
                instances = [instances]
            for inst in instances:
                webresource = inst.get("webresource", {})
                url = webresource.get("url", {}).get("$", "") if isinstance(webresource, dict) else ""
                if url:
                    candidate_urls.append(url)
            inferred_type = "thesis" if re.search(r"\bthesis\b|\bdissertation\b", title, re.I) else "research"
            if doc_type != "all" and doc_type != inferred_type:
                continue
            p = Paper(title=title, source="OpenAIRE", doc_type=inferred_type,
                       landing_url=candidate_urls[0] if candidate_urls else "")
            for u in candidate_urls:
                p.add_candidate_url(u)
            papers.append(p)
    except Exception as e:
        st.session_state.setdefault("errors", []).append(f"OpenAIRE: {e}")
    return papers


def enrich_with_unpaywall(paper, email="research-tool@example.com"):
    """Add Unpaywall's OA locations as extra candidate URLs, extending the fallback chain."""
    if not paper.doi:
        return paper
    try:
        r = requests.get(f"https://api.unpaywall.org/v2/{paper.doi}",
                          params={"email": email}, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            paper.is_oa = bool(data.get("is_oa", paper.is_oa))
            locations = []
            if data.get("best_oa_location"):
                locations.append(data["best_oa_location"])
            locations.extend(data.get("oa_locations", []) or [])
            for loc in locations:
                paper.add_candidate_url(loc.get("url_for_pdf") or loc.get("url"))
    except Exception:
        pass
    return paper


def enrich_with_core_lookup(paper, api_key=None):
    """Last-resort alternate OA location if nothing else has produced a candidate URL."""
    if paper.oa_urls or not (paper.doi or paper.title):
        return paper
    try:
        headers = dict(HEADERS)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        q = f'doi:"{paper.doi}"' if paper.doi else _build_core_query(paper.title)
        r = requests.get("https://api.core.ac.uk/v3/search/works",
                          params={"q": q, "limit": 1}, headers=headers, timeout=DEFAULT_TIMEOUT)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                w = results[0]
                download_url = w.get("downloadUrl") or w.get("download_url") or ""
                paper.add_candidate_url(download_url)
                for alt in (w.get("sourceFulltextUrls") or w.get("source_fulltext_urls") or []):
                    paper.add_candidate_url(alt)
    except Exception:
        pass
    return paper


# --------------------------------------------------------------------------
# Dedup
# --------------------------------------------------------------------------

def dedup_papers(papers):
    seen = {}
    for p in papers:
        k = p.key()
        if k not in seen:
            seen[k] = p
        else:
            existing = seen[k]
            for u in p.oa_urls:
                existing.add_candidate_url(u)
            if not existing.pages and p.pages:
                existing.pages = p.pages
    return list(seen.values())


# --------------------------------------------------------------------------
# Fallback document-to-PDF conversion
#
# When a candidate "OA" URL turns out to be an HTML page, Word/PowerPoint
# document, plain text, or EPUB rather than a PDF, we try to convert it
# instead of skipping it outright. These conversions are text-reflow only
# (not pixel-faithful to the original layout) but preserve the content,
# which is what matters for a merged reading/reference PDF.
# --------------------------------------------------------------------------

def _text_to_pdf_bytes(title, paragraphs):
    """Render a list of plain-text paragraphs into a simple PDF via reportlab."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER,
                             leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                             topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    styles = getSampleStyleSheet()
    body_style = styles["BodyText"]
    body_style.alignment = TA_LEFT
    title_style = styles["Title"]
    story = [Paragraph(_xml_escape(title or "Converted document")[:200], title_style), Spacer(1, 0.25 * inch)]
    for para in paragraphs:
        para = (para or "").strip()
        if not para:
            continue
        story.append(Paragraph(_xml_escape(para), body_style))
        story.append(Spacer(1, 0.12 * inch))
    if len(story) <= 2:
        story.append(Paragraph("(No extractable text content found.)", body_style))
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


def _convert_html_to_pdf(content, title_hint=""):
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        title = title_hint or (soup.title.string if soup.title and soup.title.string else "Converted document")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "li", "h1", "h2", "h3"])]
        paragraphs = [p for p in paragraphs if p]
        if not paragraphs:
            paragraphs = [soup.get_text(" ", strip=True)[:20000]]
        return _text_to_pdf_bytes(title, paragraphs)
    except Exception:
        return None


def _convert_docx_to_pdf(content, title_hint=""):
    try:
        import docx
        doc = docx.Document(io.BytesIO(content))
        paragraphs = [p.text for p in doc.paragraphs]
        return _text_to_pdf_bytes(title_hint or "Converted Word document", paragraphs)
    except Exception:
        return None


def _convert_pptx_to_pdf(content, title_hint=""):
    try:
        from pptx import Presentation
        prs = Presentation(io.BytesIO(content))
        paragraphs = []
        for i, slide in enumerate(prs.slides, start=1):
            paragraphs.append(f"--- Slide {i} ---")
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for p in shape.text_frame.paragraphs:
                        text = "".join(run.text for run in p.runs)
                        if text.strip():
                            paragraphs.append(text)
        return _text_to_pdf_bytes(title_hint or "Converted PowerPoint", paragraphs)
    except Exception:
        return None


def _convert_txt_to_pdf(content, title_hint=""):
    try:
        text = content.decode("utf-8", errors="ignore")
        paragraphs = text.split("\n")
        return _text_to_pdf_bytes(title_hint or "Converted text file", paragraphs)
    except Exception:
        return None


def _convert_epub_to_pdf(content, title_hint=""):
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
        book = epub.read_epub(io.BytesIO(content))
        title = title_hint
        try:
            meta = book.get_metadata("DC", "title")
            if meta:
                title = meta[0][0]
        except Exception:
            pass
        paragraphs = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                paragraphs.extend(p.get_text(" ", strip=True) for p in soup.find_all(["p", "h1", "h2", "h3"]))
        paragraphs = [p for p in paragraphs if p]
        return _text_to_pdf_bytes(title or "Converted EPUB", paragraphs)
    except Exception:
        return None


def _guess_doc_kind(url, content_type, content_head):
    """Best-effort classification of a non-PDF response we can attempt to convert."""
    lower_url = (url or "").lower()
    ctype = (content_type or "").lower()
    if lower_url.endswith((".html", ".htm")) or "html" in ctype:
        return "html"
    if lower_url.endswith(".docx") or "wordprocessingml" in ctype:
        return "docx"
    if lower_url.endswith(".doc") and "word" in ctype:
        return "doc"  # old binary .doc — not supported by python-docx, will fail gracefully
    if lower_url.endswith(".pptx") or "presentationml" in ctype:
        return "pptx"
    if lower_url.endswith(".ppt") and "powerpoint" in ctype:
        return "ppt"  # old binary .ppt — not supported, will fail gracefully
    if lower_url.endswith(".txt") or ctype.startswith("text/plain"):
        return "txt"
    if lower_url.endswith(".epub") or "epub" in ctype:
        return "epub"
    # sniff by content if extension/content-type were ambiguous
    head = content_head[:512].lstrip().lower()
    if head.startswith(b"<!doctype html") or b"<html" in head:
        return "html"
    return None


def try_convert_to_pdf(content, url, content_type, title_hint=""):
    kind = _guess_doc_kind(url, content_type, content)
    if kind == "html":
        return _convert_html_to_pdf(content, title_hint)
    if kind == "docx":
        return _convert_docx_to_pdf(content, title_hint)
    if kind == "pptx":
        return _convert_pptx_to_pdf(content, title_hint)
    if kind == "txt":
        return _convert_txt_to_pdf(content, title_hint)
    if kind == "epub":
        return _convert_epub_to_pdf(content, title_hint)
    return None  # old binary .doc/.ppt or unrecognized — no reliable pure-python path


# --------------------------------------------------------------------------
# Download with retries + alternate-location fallback
# --------------------------------------------------------------------------

@dataclass
class DownloadResult:
    paper: object
    content: bytes = None
    ok: bool = False
    reason: str = ""
    used_url: str = ""
    size_bytes: int = 0


def _attempt_single_url(url, max_size_bytes, title_hint="", timeout=30):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True, stream=True)
    except requests.exceptions.Timeout:
        return False, None, "timeout"
    except requests.exceptions.RequestException:
        return False, None, "server_error"

    if r.status_code == 404:
        return False, None, "404"
    if r.status_code == 403:
        return False, None, "403"
    if 500 <= r.status_code < 600:
        return False, None, "server_error"
    if r.status_code != 200:
        return False, None, "server_error"

    content_length = r.headers.get("Content-Length")
    if content_length and max_size_bytes and int(content_length) > max_size_bytes:
        return False, None, "too_large"

    chunks = []
    total = 0
    for chunk in r.iter_content(chunk_size=65536):
        total += len(chunk)
        if max_size_bytes and total > max_size_bytes:
            return False, None, "too_large"
        chunks.append(chunk)
    content = b"".join(chunks)

    ctype = r.headers.get("Content-Type", "").lower()
    if content[:5] != b"%PDF-" and "pdf" not in ctype:
        # Not a PDF — try converting it (HTML/DOCX/PPTX/TXT/EPUB) before giving up on this URL.
        converted = try_convert_to_pdf(content, url, ctype, title_hint)
        if converted:
            if max_size_bytes and len(converted) > max_size_bytes:
                return False, None, "too_large"
            try:
                PdfReader(io.BytesIO(converted))
            except Exception:
                return False, None, "not_pdf"
            return True, converted, "ok_converted"
        return False, None, "not_pdf"

    try:
        PdfReader(io.BytesIO(content))
    except Exception:
        return False, None, "invalid_pdf"

    return True, content, "ok"


def download_with_fallback(paper, max_size_mb=None, retries=MAX_RETRIES_PER_URL):
    """Try each candidate URL in order; retry each URL on transient failures
    before moving to the next candidate (alternate OA location)."""
    max_size_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb else None
    if not paper.oa_urls:
        return DownloadResult(paper=paper, ok=False, reason="no_candidates")

    last_reason = "no_candidates"
    for url in paper.oa_urls:
        for attempt in range(1, retries + 1):
            ok, content, reason = _attempt_single_url(url, max_size_bytes, title_hint=paper.title)
            if ok:
                return DownloadResult(paper=paper, content=content, ok=True,
                                       reason=reason, used_url=url, size_bytes=len(content))
            last_reason = reason
            if reason in ("timeout", "server_error"):
                if attempt < retries:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
            break  # definitive failure (404 / 403 / not_pdf / invalid_pdf / too_large) -> next URL
    return DownloadResult(paper=paper, ok=False, reason=last_reason)


# --------------------------------------------------------------------------
# Output builders: merged PDF, CSV, TXT bibliography
# --------------------------------------------------------------------------

def make_toc_page(papers):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER
    margin = 0.75 * inch
    y = height - margin
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Table of Contents")
    y -= 0.4 * inch
    c.setFont("Helvetica", 9)
    for i, p in enumerate(papers, start=1):
        line = f"{i}. {p.citation_line()} [{p.source}]"
        max_chars = 110
        chunks = [line[j:j + max_chars] for j in range(0, len(line), max_chars)] or [line]
        for j, chunk in enumerate(chunks):
            prefix = "" if j == 0 else "    "
            c.drawString(margin, y, prefix + chunk)
            y -= 0.2 * inch
            if y < margin:
                c.showPage()
                c.setFont("Helvetica", 9)
                y = height - margin
        y -= 0.05 * inch
        if y < margin:
            c.showPage()
            c.setFont("Helvetica", 9)
            y = height - margin
    c.save()
    buf.seek(0)
    return buf


def merge_pdfs(toc_buf, pdf_items, progress_callback=None):
    writer = PdfWriter()
    toc_reader = PdfReader(toc_buf)
    for pg in toc_reader.pages:
        writer.add_page(pg)
    page_cursor = len(toc_reader.pages)
    total = len(pdf_items) or 1
    for i, (paper, content) in enumerate(pdf_items, start=1):
        try:
            reader = PdfReader(io.BytesIO(content))
            start_page = page_cursor
            for pg in reader.pages:
                writer.add_page(pg)
            writer.add_outline_item(paper.title[:100] or "Untitled", start_page)
            page_cursor += len(reader.pages)
        except Exception:
            pass
        if progress_callback:
            progress_callback(i / total, paper.title)
    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out


def build_metadata_txt(downloaded_papers, skipped_results, keyword):
    lines = []
    lines.append("PAPER MASS DOWNLOAD — METADATA & BIBLIOGRAPHY")
    lines.append(f"Search keyword: {keyword}")
    lines.append(f"Included: {len(downloaded_papers)}   Skipped: {len(skipped_results)}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("TITLES")
    lines.append("-" * 70)
    for i, p in enumerate(downloaded_papers, start=1):
        lines.append(f"{i}. {p.title}  ({p.year or 'n.d.'}) [{p.doc_type or 'research'}, {p.source}]")
    lines.append("")
    lines.append("FULL METADATA")
    lines.append("-" * 70)
    for i, p in enumerate(downloaded_papers, start=1):
        lines.append(f"[{i}] {p.title}")
        lines.append(f"    Author(s): {', '.join(p.authors) or 'Unknown'}")
        lines.append(f"    Year: {p.year or 'n/a'}    Type: {p.doc_type or 'research'}    Source: {p.source}")
        lines.append(f"    Venue: {p.venue or 'n/a'}    Pages: {p.pages or 'n/a'}")
        lines.append(f"    DOI: {p.doi or 'n/a'}")
        lines.append(f"    Open Access: {'Yes' if p.is_oa else 'Unknown'}")
        lines.append("")
    lines.append("BIBLIOGRAPHY")
    lines.append("-" * 70)
    for i, p in enumerate(downloaded_papers, start=1):
        lines.append(p.bibliography_entry(i))
    if skipped_results:
        lines.append("")
        lines.append("SKIPPED (no usable OA PDF found)")
        lines.append("-" * 70)
        for r in skipped_results:
            lines.append(f"- {r.paper.title}  [reason: {r.reason}]")
    return "\n".join(lines).encode("utf-8")


def build_csv(downloaded_papers):
    lines = ["Title,Authors,Year,Type,Source,DOI,Pages,OA,Used_URL"]
    for p, used_url in downloaded_papers:
        lines.append(
            f'"{p.title}","{"; ".join(p.authors)}",{p.year},{p.doc_type},{p.source},'
            f'{p.doi},{p.pages},{"Yes" if p.is_oa else "Unknown"},{used_url}'
        )
    return "\n".join(lines).encode("utf-8")


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("📚 " + APP_TITLE)
st.caption(
    "Searches OpenAlex, Crossref, Unpaywall, CORE, DOAJ and OpenAIRE for legally open-access "
    "papers and theses, then merges your selection into a single downloadable PDF. "
    "Google Scholar is excluded — it has no public API and blocks automated access."
)

if "search_stage" not in st.session_state:
    st.session_state["search_stage"] = "initial"

with st.form("search_form"):
    col1, col2 = st.columns(2)
    with col1:
        keyword = st.text_input("Keyword(s) (optional if a field is specified)", placeholder="e.g. enhanced oil recovery")
        field_name = st.text_input("Field / subject (optional if a keyword is specified)", placeholder="e.g. Petroleum Engineering")
        country = st.text_input("Country code (optional, ISO 2-letter)", placeholder="e.g. TR, US, DE")
        institution = st.text_input("University / institution (optional)", placeholder="e.g. Istanbul Technical University")
        use_query_expansion = st.checkbox(
            "Suggest expanded search terms from initial results before the full search",
            value=False,
            help="Runs a quick first-pass search, extracts common terminology from the "
                 "titles found, and lets you pick expanded queries to broaden the real search.",
        )
    with col2:
        year_from, year_to = st.slider("Publication year range", 1990, 2026, (2015, 2026))
        max_results = st.number_input("Max number of PDFs to include", min_value=1, max_value=200, value=20, step=1)
        doc_type_choice = st.radio(
            "Document type",
            ["Research papers", "Theses / dissertations", "Conference papers", "Reports", "All"],
            help="Theses live in different repositories than journal databases, so this is "
                 "treated as its own category rather than lumped in with research papers.",
        )
        max_pdf_size_mb = st.number_input(
            "Maximum PDF size per file (MB, 0 = no limit)", min_value=0.0, value=0.0, step=1.0,
            help="Files larger than this are skipped rather than downloaded.",
        )
        sources = st.multiselect(
            "Sources to search",
            ["OpenAlex", "Crossref", "CORE", "DOAJ", "OpenAIRE"],
            default=["OpenAlex", "Crossref", "CORE", "DOAJ"],
            help="DOAJ indexes journals only and is skipped automatically outside 'Research papers'/'All'. "
                 "CORE works without a key but a free key (core.ac.uk/services/api) raises its daily "
                 "rate limit a lot and is needed for full-text search.",
        )
        core_key = st.text_input(
            "CORE API key (optional — raises CORE's rate limit; get a free one at core.ac.uk/services/api)",
            type="password",
        )
        show_core_debug = st.checkbox(
            "Show CORE debug info after searching (raw HTTP status/response)",
            value=False,
            help="Useful for diagnosing why CORE returns 0 results — shows the exact "
                 "query sent and CORE's raw response.",
        )

    submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

DOC_TYPE_VALUE = {
    "Research papers": "research",
    "Theses / dissertations": "thesis",
    "Conference papers": "conference",
    "Reports": "report",
    "All": "all",
}


def run_full_search(keyword_str, year_from, year_to, country, institution, field_name,
                     max_results, doc_type, sources, core_key, progress=None):
    st.session_state["errors"] = []
    st.session_state["core_debug"] = []
    per_source_limit = max(max_results * 4, 40)
    all_papers = []

    # Fixed set of stages so the progress bar has a stable denominator:
    # one step per source being queried, plus enrichment steps at the end.
    total_steps = len(sources) + 2  # + unpaywall enrichment + core fallback enrichment
    step = 0

    def _tick(label):
        nonlocal step
        step += 1
        if progress is not None:
            progress.progress(min(step / total_steps, 1.0), text=label)

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {}
        if "OpenAlex" in sources:
            futures[ex.submit(search_openalex, keyword_str, year_from, year_to, country, institution, field_name, per_source_limit, doc_type)] = "OpenAlex"
        if "Crossref" in sources:
            futures[ex.submit(search_crossref, keyword_str, year_from, year_to, per_source_limit, doc_type)] = "Crossref"
        if "CORE" in sources:
            futures[ex.submit(search_core, keyword_str, year_from, year_to, per_source_limit, core_key or None, doc_type)] = "CORE"
        if "DOAJ" in sources:
            futures[ex.submit(search_doaj, keyword_str, year_from, year_to, per_source_limit, doc_type)] = "DOAJ"
        if "OpenAIRE" in sources:
            futures[ex.submit(search_openaire, keyword_str, year_from, year_to, per_source_limit, doc_type)] = "OpenAIRE"
        for fut in as_completed(futures):
            src_name = futures[fut]
            try:
                all_papers.extend(fut.result())
            except Exception as e:
                st.session_state["errors"].append(f"{src_name}: {e}")
            _tick(f"Searched {src_name}...")

    if doc_type != "all":
        all_papers = [p for p in all_papers if (p.doc_type or "research") == doc_type]

    deduped = dedup_papers(all_papers)
    if field_name:
        for p in deduped:
            if not p.subject_field:
                p.subject_field = field_name

    with ThreadPoolExecutor(max_workers=10) as ex:
        deduped = list(ex.map(enrich_with_unpaywall, deduped))
    _tick("Cross-checking Unpaywall for extra OA links...")

    with ThreadPoolExecutor(max_workers=5) as ex:
        deduped = list(ex.map(lambda p: enrich_with_core_lookup(p, core_key or None), deduped))
    _tick("Filling in remaining gaps via CORE...")

    return [p for p in deduped if p.oa_urls]


if submitted:
    if not keyword.strip() and not field_name.strip():
        st.error("Please enter at least a keyword or a field/subject.")
        st.stop()

    search_term = keyword.strip() or field_name.strip()

    doc_type = DOC_TYPE_VALUE[doc_type_choice]
    st.session_state["form_values"] = dict(
        keyword=search_term, field_name=field_name, country=country, institution=institution,
        year_from=year_from, year_to=year_to, max_results=max_results, doc_type=doc_type,
        max_pdf_size_mb=max_pdf_size_mb, sources=sources, core_key=core_key,
        show_core_debug=show_core_debug,
    )

    if use_query_expansion:
        st.write("**Initial pass** — learning common terminology from early results:")
        first_pass_progress = st.progress(0.0, text="Starting initial search...")
        first_pass = run_full_search(search_term, year_from, year_to, country, institution,
                                      field_name, min(max_results, 15), doc_type, sources, core_key,
                                      progress=first_pass_progress)
        first_pass_progress.progress(1.0, text="Initial pass complete.")
        terms = extract_keywords(first_pass)
        suggestions = suggest_expanded_queries(search_term, terms)
        st.session_state["expansion_first_pass"] = first_pass
        st.session_state["expansion_suggestions"] = suggestions
        st.session_state["search_stage"] = "suggested"
        st.session_state.pop("candidates", None)
    else:
        st.write("**Querying scholarly databases:**")
        search_progress = st.progress(0.0, text="Starting search...")
        candidates = run_full_search(search_term, year_from, year_to, country, institution,
                                      field_name, max_results, doc_type, sources, core_key,
                                      progress=search_progress)
        search_progress.progress(1.0, text="Search complete.")
        st.session_state["candidates"] = candidates
        st.session_state["target_count"] = max_results
        st.session_state["search_stage"] = "final"
        st.success(f"Found {len(candidates)} unique matches with a resolvable open-access candidate link.")
        if st.session_state.get("errors"):
            st.warning("Some sources didn't return results:")
            for e in st.session_state["errors"]:
                st.write("-", e)
        if show_core_debug and st.session_state.get("core_debug"):
            with st.expander("🔍 CORE debug info"):
                for item in st.session_state["core_debug"]:
                    st.json(item)

# --------------------------------------------------------------------------
# Query expansion step
# --------------------------------------------------------------------------

if st.session_state.get("search_stage") == "suggested":
    st.subheader("Suggested expanded searches")
    st.caption(
        "Based on common terminology found in your initial results, here are some "
        "broadened queries. Pick as many as you like, or stick with your original keyword."
    )
    fv = st.session_state["form_values"]
    suggestions = st.session_state.get("expansion_suggestions", [])
    chosen = st.multiselect(
        "Additional query variants to include",
        suggestions,
        default=suggestions[:3],
    )
    run_original_too = st.checkbox(f"Also include the original query: \"{fv['keyword']}\"", value=True)

    if st.button("Run expanded search", type="primary"):
        queries = ([fv["keyword"]] if run_original_too else []) + chosen
        if not queries:
            st.error("Select at least one query to run.")
        else:
            all_candidates = []
            st.write(f"**Searching {len(queries)} query variant(s):**")
            overall_progress = st.progress(0.0, text="Starting expanded search...")
            for qi, q in enumerate(queries):
                sub_progress_placeholder = st.empty()
                sub_progress = sub_progress_placeholder.progress(
                    0.0, text=f"Query {qi + 1}/{len(queries)}: \"{q}\"..."
                )
                all_candidates.extend(run_full_search(
                    q, fv["year_from"], fv["year_to"], fv["country"], fv["institution"],
                    fv["field_name"], fv["max_results"], fv["doc_type"], fv["sources"], fv["core_key"],
                    progress=sub_progress
                ))
                sub_progress_placeholder.empty()
                overall_progress.progress((qi + 1) / len(queries),
                                           text=f"Completed {qi + 1}/{len(queries)} quer(y/ies)...")
            overall_progress.progress(1.0, text="Expanded search complete.")
            deduped = dedup_papers(all_candidates)
            st.session_state["candidates"] = deduped
            st.session_state["target_count"] = fv["max_results"]
            st.session_state["search_stage"] = "final"
            st.success(f"Found {len(deduped)} unique matches across {len(queries)} quer(y/ies).")
            if fv.get("show_core_debug") and st.session_state.get("core_debug"):
                with st.expander("🔍 CORE debug info"):
                    for item in st.session_state["core_debug"]:
                        st.json(item)
            st.rerun()

# --------------------------------------------------------------------------
# Results + selection + download
# --------------------------------------------------------------------------

if st.session_state.get("candidates") and st.session_state.get("search_stage") == "final":
    candidates = st.session_state["candidates"]
    target = st.session_state["target_count"]
    fv = st.session_state.get("form_values", {})
    max_pdf_size_mb = fv.get("max_pdf_size_mb", 0.0)

    st.subheader(f"Select papers to include ({len(candidates)} found with a resolvable PDF candidate)")
    st.caption(f"The top {min(target, len(candidates))} are pre-ticked based on your requested count — "
               "tick or untick rows to change the selection, then download.")

    table_rows = [{
        "Include": i < target,
        "Title": p.title or "(untitled)",
        "Year": p.year,
        "Type": p.doc_type or "research",
        "OA": "Yes" if p.is_oa else "Unknown",
        "DOI": p.doi,
        "Pages": p.pages,
        "Source": p.source,
    } for i, p in enumerate(candidates)]

    edited = st.data_editor(
        table_rows,
        column_config={
            "Include": st.column_config.CheckboxColumn("Include", default=False, width="small"),
            "Title": st.column_config.TextColumn("Title", width="large"),
        },
        disabled=["Title", "Year", "Type", "OA", "DOI", "Pages", "Source"],
        hide_index=True,
        use_container_width=True,
        height=min(60 + 35 * len(candidates), 600),
        key="selection_editor",
    )

    selected_indices = [i for i, row in enumerate(edited) if row["Include"]]
    selected_papers = [candidates[i] for i in selected_indices]
    st.write(f"**{len(selected_papers)} paper(s) selected.**"
             + (f"  Max size per PDF: {max_pdf_size_mb} MB." if max_pdf_size_mb else "  No size limit set."))

    if st.button(f"⬇️ Download & merge {len(selected_papers)} selected PDFs",
                 type="primary", disabled=not selected_papers):
        progress = st.progress(0.0, text="Starting downloads...")
        results = []
        total_to_try = len(selected_papers)

        with ThreadPoolExecutor(max_workers=6) as ex:
            future_map = {
                ex.submit(download_with_fallback, p, max_pdf_size_mb or None): p
                for p in selected_papers
            }
            done = 0
            for fut in as_completed(future_map):
                done += 1
                progress.progress(min(done / total_to_try, 1.0), text=f"Checked {done}/{total_to_try}...")
                results.append(fut.result())

        downloaded = [(r.paper, r.content) for r in results if r.ok]
        downloaded_with_url = [(r.paper, r.used_url) for r in results if r.ok]
        skipped_results = [r for r in results if not r.ok]

        if not downloaded:
            st.error("Couldn't download any PDFs from the selection — try broadening filters, "
                      "raising the size limit, or adding more sources.")
        else:
            with st.spinner("Building table of contents..."):
                toc_buf = make_toc_page([p for p, _ in downloaded])

            merge_progress = st.progress(0.0, text="Merging PDFs...")

            def _merge_progress_cb(frac, title):
                merge_progress.progress(frac, text=f"Merging: {title[:60]}...")

            merged = merge_pdfs(toc_buf, downloaded, progress_callback=_merge_progress_cb)
            merge_progress.progress(1.0, text="Merge complete.")

            reason_counts = {}
            for r in skipped_results:
                reason_counts[r.reason] = reason_counts.get(r.reason, 0) + 1
            reason_summary = ", ".join(f"{v}× {k}" for k, v in reason_counts.items())

            st.success(f"Merged {len(downloaded)} PDFs into one file. "
                       f"{len(skipped_results)} skipped" + (f" ({reason_summary})." if reason_summary else "."))
            converted_count = sum(1 for r in results if r.reason == "ok_converted")
            if converted_count:
                st.caption(f"{converted_count} file(s) weren't natively PDF and were converted "
                           "(HTML/DOCX/PPTX/TXT/EPUB → text-reflow PDF) before merging.")

            colA, colB, colC = st.columns(3)
            with colA:
                st.download_button(
                    "📥 Merged PDF",
                    data=merged,
                    file_name=f"papers_{fv.get('keyword','search').replace(' ', '_')[:30]}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )
            with colB:
                st.download_button(
                    "📄 Metadata (CSV)",
                    data=build_csv(downloaded_with_url),
                    file_name="papers_metadata.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with colC:
                st.download_button(
                    "📝 Titles + bibliography (TXT)",
                    data=build_metadata_txt([p for p, _ in downloaded], skipped_results, fv.get("keyword", "")),
                    file_name="papers_bibliography.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            if skipped_results:
                with st.expander(f"Why {len(skipped_results)} paper(s) were skipped"):
                    for r in skipped_results:
                        st.write(f"- **{r.paper.title[:80]}** — {r.reason}")

st.divider()
st.caption(
    "Only papers with a legal open-access PDF (via OpenAlex, Unpaywall, CORE, DOAJ or OpenAIRE) are downloaded. "
    "Non-PDF OA files (HTML, DOCX, PPTX, TXT, EPUB) are converted to PDF automatically before being counted as "
    "skipped — old binary .doc/.ppt formats can't be reliably converted without external tools and are skipped. "
    "Each paper's candidate OA locations are tried in sequence with retries on transient errors before being "
    "marked as skipped. Paywalled papers without any OA copy are skipped, not bypassed."
)
