import concurrent.futures
import json
import re
from enum import Enum
from pathlib import Path
from typing import List, Tuple

import requests
import spacy
import typer
from bs4 import BeautifulSoup
from spacy.matcher import Matcher

__all__ = ["crawler"]

crawler = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


"""
# Get links from arxiv
deep-learning-at-scale chapter_3 crawler get-links

# Extract features 

time deep-learning-at-scale chapter_3 crawler extract-sentences --mode process
# deep-learning-at-scale chapter_3 crawler extract-sentences --mode process  

time deep-learning-at-scale chapter_3 crawler extract-sentences --mode threaded
# deep-learning-at-scale chapter_3 crawler extract-sentences --mode threaded  

time deep-learning-at-scale chapter_3 crawler extract-sentences
# deep-learning-at-scale chapter_3 crawler extract-sentences  


### With 5 workers 

# time deep-learning-at-scale chapter_3 crawler extract-sentences
# deep-learning-at-scale chapter_3 crawler extract-sentences

# time deep-learning-at-scale chapter_3 crawler extract-sentences --mode threaded
# deep-learning-at-scale chapter_3 crawler extract-sentences --mode threaded

# time deep-learning-at-scale chapter_3 crawler extract-sentences --mode process
# deep-learning-at-scale chapter_3 crawler extract-sentences --mode process
"""


class ExecMode(str, Enum):
    Serial = "Serial"
    Threaded = "Threaded"
    Process = "Process"


class TextMatcher:
    def __init__(
        self,
        phase_pattern: str = "deep[ -]learning",
        full_name_pattern: List[str] = [{"POS": "PROPN"}, {"POS": "PROPN"}],
    ) -> None:
        self.phase_pattern: str = phase_pattern
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add("FULL_NAME", [full_name_pattern])

    def find_matched_sentences(self, url: str) -> Tuple[List[str], List[str]]:
        about_doc = self._extract_visible_from_url(url)
        sentences = self._extract_sentence_with_pattern(about_doc)
        names = self._extract_full_names(about_doc)
        return sentences, names

    def find_matched_sentences_batch(
        self, urls: List[str]
    ) -> Tuple[List[str], List[str]]:
        all_sentences, all_names = set(), set()
        for link in urls:
            sentences, names = self.find_matched_sentences(link)
            all_sentences.update(sentences)
            all_names.update(names)
        return list(all_sentences), list(all_names)

    def _extract_full_names(self, about_doc) -> List[str]:
        matches = self.matcher(about_doc)
        names = []
        for _, start, end in matches:
            span = about_doc[start:end]
            names.append(span.text)
        return names

    def _extract_visible_from_url(self, url: str) -> str:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        texts = soup.findAll(text=True)
        texts = " ".join(t.strip() for t in texts)
        about_doc = self.nlp(texts)
        return about_doc

    def _extract_sentence_with_pattern(self, about_doc: str) -> List[str]:
        return re.findall(
            rf"[^.]*?{self.phase_pattern}[^.]*?\.",
            ".".join([f.text for f in about_doc.sents]),
            flags=re.IGNORECASE,
        )


class LinkFinder:
    def __init__(
        self,
        seed_url: str,
        base_url: str,
        base_url_constraint: str,
        max_results: int = 100,
    ) -> None:
        self.seed_url = seed_url
        self.base_url = base_url
        self.base_url_constraint = base_url_constraint
        self.max_results = max_results

    def _extract_links(self, elements):
        links = []
        for e in elements:
            url = e["href"]
            if "https://" not in url:
                url = self.base_url + url
            if self.base_url in url and url.startswith(self.base_url_constraint):
                links.append(url)
        return set(links[: self.max_results]) if self.max_results != -1 else set(links)

    def find_links(self, start_url: str = None):
        url = start_url if start_url else self.seed_url
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        elements = soup.find_all("a", href=True)
        links = self._extract_links(elements)

        # for url in links:
        #     new_links = self.find_links(url)
        #     links = links.union(new_links)
        return links


def serial_extract(links: List[str], phase_pattern: str):
    sentence_matcher = TextMatcher(phase_pattern=phase_pattern)
    all_sentences, all_names = set(), set()
    for link in links:
        sentences, names = sentence_matcher.find_matched_sentences(link)
        all_sentences.update(sentences)
        all_names.update(names)
    return list(all_sentences), list(all_names)


def parallelised_sentences(
    links, phase_pattern: str, max_workers: int = 5, use_process: bool = False
):
    executor = (
        concurrent.futures.ProcessPoolExecutor
        if use_process
        else concurrent.futures.ThreadPoolExecutor
    )
    all_sentences, all_names = set(), set()
    with executor(max_workers=max_workers) as executor:
        url_future = {
            executor.submit(
                TextMatcher(phase_pattern=phase_pattern).find_matched_sentences, url
            ): url
            for url in links
        }
        for future in concurrent.futures.as_completed(url_future):
            sentences, names = future.result()
            all_sentences.update(sentences)
            all_names.update(names)
    return list(all_sentences), list(all_names)


@crawler.command()
def get_links(
    base_url: str = typer.Option("https://arxiv.org/"),
    start_suffix: str = typer.Option("list/cs.LG/pastweek?show=1063"),
    filter_suffix: str = typer.Option("/abs"),
    output_fn: Path = typer.Option(Path("links.json")),
):
    start_url: str = base_url + start_suffix
    filter_base_url: str = base_url + filter_suffix
    link_finder = LinkFinder(
        seed_url=start_url, base_url=base_url, base_url_constraint=filter_base_url
    )

    links = link_finder.find_links()
    with open(output_fn, "w") as f:
        json.dump(list(links), f)


@crawler.command()
def extract_sentences(
    mode: ExecMode = typer.Option(ExecMode.Serial, case_sensitive=False),
    links_fn: Path = typer.Option(Path("links.json")),
    phase_pattern: str = typer.Option("deep[ -]learning"),
    max_workers: int = typer.Option(10),
):
    with open(links_fn, "rb") as f:
        links = json.load(f)

    sentences, names = [], []
    if mode == ExecMode.Serial:
        sentences, names = serial_extract(links, phase_pattern)
    elif mode == ExecMode.Threaded:
        sentences, names = parallelised_sentences(
            links, phase_pattern, use_process=False
        )
    elif mode == ExecMode.Process:
        sentences, names = parallelised_sentences(
            links, phase_pattern, use_process=True, max_workers=max_workers
        )
    else:
        raise ValueError("No such mode known")

    with open("resullts.json", "w") as f:
        json.dump({"sentences": sentences, "names": names}, f)
