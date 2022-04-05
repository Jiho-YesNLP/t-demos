"""
Neomodel documentation: https://neomodel.readthedocs.io/en/latest/

how to:
- delete everything from db?
    match (a)-[r]->() delete a, r; match(a) delete a;
"""
import code
import os
import gzip
import hashlib

from tqdm import tqdm
from lxml import etree
from dotenv import load_dotenv
load_dotenv()

from neomodel import StructuredNode, UniqueIdProperty, StringProperty, \
                     RelationshipTo, config

config.DATABASE_URL = \
    f"bolt://neo4j:{os.environ['NEO4J_PASSWORD']}@localhost:7687"

MAPPING = {
    'PMID': 'PMID',
    'paper_title': 'Article/ArticleTitle',
    'paper_text': 'Article/Abstract/AbstractText',
    'ISSN': 'Article/Journal/ISSN',
    'journal_title': 'Article/Journal/Title',
    'author': 'Article/AuthorList/Author',
    'keyword': 'KeywordList/Keyword',
    'mesh': 'MeshHeadingList/MeshHeading/DescriptorName'
}

class Journal(StructuredNode):
    ISSN = UniqueIdProperty()
    title = StringProperty()
    ISO_abbrev = StringProperty()

class Author(StructuredNode):
    author_id = UniqueIdProperty()
    full_name = StringProperty()

class Keyword(StructuredNode):
    kw_id = UniqueIdProperty()
    term = StringProperty()

class MeSH(StructuredNode):
    mesh_ui = UniqueIdProperty()
    term = StringProperty()

class Paper(StructuredNode):
    PMID = UniqueIdProperty()
    paper_title = StringProperty()
    paper_text = StringProperty()

    # relationships
    journal = RelationshipTo(Journal, 'published_by')
    author = RelationshipTo(Author, 'written_by')
    keyword = RelationshipTo(Keyword, 'contains')
    mesh = RelationshipTo(MeSH, 'contains')


if __name__ == '__main__':
    # Read and parse all Medline archive files
    source_file = 'pubmed22n1081.xml.gz'
    pbar = tqdm(total=30000)  # rough estimate
    with gzip.open(source_file) as f:
        root = etree.parse(f).getroot()
        for cnt, c in enumerate(root.iterfind('.//MedlineCitation')):
            PMID = c.findtext('PMID')

            Paper
            entry = {k: c.findtext(MAPPING[k]) 
                     for k in ['PMID', 'paper_title', 'paper_text']}
            p = Paper.get_or_create(entry)[0]

            # Journal
            entry = {k: c.findtext(MAPPING[k]) 
                     for k in ['ISSN', 'journal_title']}
            j = Journal.get_or_create(entry)[0]
            p.journal.connect(j)
                
            # Authors
            for a in c.iterfind(MAPPING['author']):
                first = a.findtext('ForeName')
                last = a.findtext('LastName')
                init = a.findtext('Initials')
                name = f'{first}, {last}' if init is None or len(init) == 0 \
                    else f'{first} {init}, {last}'
                author_id = hashlib.md5(name.encode('utf-8')).hexdigest()[:8]
                entry = {'author_id': author_id, 'full_name': name}
                a = Author.get_or_create(entry)[0]
                p.author.connect(a)

            # Keywords
            for k in c.findall(MAPPING['keyword']):
                if k.text is None:
                    continue
                kw_id = hashlib.md5(k.text.encode('utf-8')).hexdigest()[:8]
                entry = {'kw_id': kw_id, 'term': k.text}
                k = Keyword.get_or_create(entry)[0]
                p.keyword.connect(k)

            # MeSH
            for m in c.findall(MAPPING['mesh']):
                if m.text is None:
                    continue
                entry = {'mesh_ui': m.get('UI'), 'term': m.text}
                m = MeSH.get_or_create(entry)[0]
                p.mesh.connect(m)
            pbar.update()
        pbar.close()