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
from pypdf import PdfWriter, PdfReader
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

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
    field: str = ""
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
                   f"to_publication_date:{year_to}-12-31",
                   "has_fulltext:true"]
        if country:
            filters.append(f"institutions.country_code:{country}")
        type_map = {"thesis": "dissertation", "research": "article",
                    "conference": "paratext|preprint", "report": "report"}
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
        r.raise_for_status()
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
                field=field_name,
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


def search_core(keyword, year_from, year_to, limit, api_key=None, doc_type="all"):
    """CORE's repository network is especially strong for theses."""
    papers = []
    try:
        headers = dict(HEADERS)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        q = f'{keyword} AND yearPublished>={year_from} AND yearPublished<={year_to}'
        type_map = {"thesis": "thesis", "research": "research", "conference": "conference proceedings"}
        if doc_type != "all" and doc_type in type_map:
            q += f' AND documentType:"{type_map[doc_type]}"'
        params = {"q": q, "limit": min(limit, 100)}
        r = requests.get("https://api.core.ac.uk/v3/search/works", params=params,
                          headers=headers, timeout=DEFAULT_TIMEOUT)
        if r.status_code == 401:
            return papers
        r.raise_for_status()
        for w in r.json().get("results", []):
            raw_type = (w.get("documentType") or "").lower()
            if "thesis" in raw_type:
                ptype = "thesis"
            elif "conference" in raw_type:
                ptype = "conference"
            elif "report" in raw_type:
                ptype = "report"
            else:
                ptype = "research"
            p = Paper(
                title=w.get("title", ""),
                authors=[a.get("name", "") for a in (w.get("authors") or [])],
                year=str(w.get("yearPublished") or ""),
                venue=(w.get("publisher") or ""),
                doi=w.get("doi", "") or "",
                source="CORE",
                landing_url=w.get("sourceFulltextUrls", [""])[0] if w.get("sourceFulltextUrls") else "",
                doc_type=ptype,
                is_oa=True,
            )
            p.add_candidate_url(w.get("downloadUrl", ""))
            for alt in (w.get("sourceFulltextUrls") or []):
                p.add_candidate_url(alt)
            papers.append(p)
    except Exception as e:
        st.session_state.setdefault("errors", []).append(f"CORE: {e}")
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
        q = f'doi:"{paper.doi}"' if paper.doi else paper.title
        r = requests.get("https://api.core.ac.uk/v3/search/works",
                          params={"q": q, "limit": 1}, headers=headers, timeout=DEFAULT_TIMEOUT)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                paper.add_candidate_url(results[0].get("downloadUrl", ""))
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


def _attempt_single_url(url, max_size_bytes, timeout=30):
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
            ok, content, reason = _attempt_single_url(url, max_size_bytes)
            if ok:
                return DownloadResult(paper=paper, content=content, ok=True,
                                       reason="ok", used_url=url, size_bytes=len(content))
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


def merge_pdfs(toc_buf, pdf_items):
    writer = PdfWriter()
    toc_reader = PdfReader(toc_buf)
    for pg in toc_reader.pages:
        writer.add_page(pg)
    page_cursor = len(toc_reader.pages)
    for paper, content in pdf_items:
        try:
            reader = PdfReader(io.BytesIO(content))
            start_page = page_cursor
            for pg in reader.pages:
                writer.add_page(pg)
            writer.add_outline_item(paper.title[:100] or "Untitled", start_page)
            page_cursor += len(reader.pages)
        except Exception:
            continue
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
        keyword = st.text_input("Keyword(s) *", placeholder="e.g. enhanced oil recovery")
        field_name = st.text_input("Field / subject (optional)", placeholder="e.g. Petroleum Engineering")
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
            default=["OpenAlex", "Crossref", "DOAJ"],
            help="DOAJ indexes journals only and is skipped automatically outside 'Research papers'/'All'.",
        )
        core_key = st.text_input("CORE API key (optional — improves CORE results, esp. for theses)", type="password")

    submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

DOC_TYPE_VALUE = {
    "Research papers": "research",
    "Theses / dissertations": "thesis",
    "Conference papers": "conference",
    "Reports": "report",
    "All": "all",
}


def run_full_search(keyword_str, year_from, year_to, country, institution, field_name,
                     max_results, doc_type, sources, core_key):
    st.session_state["errors"] = []
    per_source_limit = max(max_results * 4, 40)
    all_papers = []
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
            try:
                all_papers.extend(fut.result())
            except Exception as e:
                st.session_state["errors"].append(f"{futures[fut]}: {e}")

    if doc_type != "all":
        all_papers = [p for p in all_papers if (p.doc_type or "research") == doc_type]

    deduped = dedup_papers(all_papers)
    if field_name:
        for p in deduped:
            if not p.field:
                p.field = field_name

    with ThreadPoolExecutor(max_workers=10) as ex:
        deduped = list(ex.map(enrich_with_unpaywall, deduped))
    with ThreadPoolExecutor(max_workers=5) as ex:
        deduped = list(ex.map(lambda p: enrich_with_core_lookup(p, core_key or None), deduped))

    return [p for p in deduped if p.oa_urls]


if submitted:
    if not keyword.strip():
        st.error("Please enter at least one keyword.")
        st.stop()

    doc_type = DOC_TYPE_VALUE[doc_type_choice]
    st.session_state["form_values"] = dict(
        keyword=keyword, field_name=field_name, country=country, institution=institution,
        year_from=year_from, year_to=year_to, max_results=max_results, doc_type=doc_type,
        max_pdf_size_mb=max_pdf_size_mb, sources=sources, core_key=core_key,
    )

    if use_query_expansion:
        with st.spinner("Running an initial pass to learn common terminology..."):
            first_pass = run_full_search(keyword, year_from, year_to, country, institution,
                                          field_name, min(max_results, 15), doc_type, sources, core_key)
        terms = extract_keywords(first_pass)
        suggestions = suggest_expanded_queries(keyword, terms)
        st.session_state["expansion_first_pass"] = first_pass
        st.session_state["expansion_suggestions"] = suggestions
        st.session_state["search_stage"] = "suggested"
        st.session_state.pop("candidates", None)
    else:
        with st.spinner("Querying scholarly databases..."):
            candidates = run_full_search(keyword, year_from, year_to, country, institution,
                                          field_name, max_results, doc_type, sources, core_key)
        st.session_state["candidates"] = candidates
        st.session_state["target_count"] = max_results
        st.session_state["search_stage"] = "final"
        st.success(f"Found {len(candidates)} unique matches with a resolvable open-access candidate link.")
        if st.session_state.get("errors"):
            with st.expander("⚠️ Some sources had issues"):
                for e in st.session_state["errors"]:
                    st.write("-", e)

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
            with st.spinner(f"Searching {len(queries)} query variant(s)..."):
                for q in queries:
                    all_candidates.extend(run_full_search(
                        q, fv["year_from"], fv["year_to"], fv["country"], fv["institution"],
                        fv["field_name"], fv["max_results"], fv["doc_type"], fv["sources"], fv["core_key"]
                    ))
            deduped = dedup_papers(all_candidates)
            st.session_state["candidates"] = deduped
            st.session_state["target_count"] = fv["max_results"]
            st.session_state["search_stage"] = "final"
            st.success(f"Found {len(deduped)} unique matches across {len(queries)} quer(y/ies).")
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
            with st.spinner("Merging into a single PDF..."):
                toc_buf = make_toc_page([p for p, _ in downloaded])
                merged = merge_pdfs(toc_buf, downloaded)

            reason_counts = {}
            for r in skipped_results:
                reason_counts[r.reason] = reason_counts.get(r.reason, 0) + 1
            reason_summary = ", ".join(f"{v}× {k}" for k, v in reason_counts.items())

            st.success(f"Merged {len(downloaded)} PDFs into one file. "
                       f"{len(skipped_results)} skipped" + (f" ({reason_summary})." if reason_summary else "."))

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
    "Each paper's candidate OA locations are tried in sequence with retries on transient errors before being "
    "marked as skipped. Paywalled papers without any OA copy are skipped, not bypassed."
)
