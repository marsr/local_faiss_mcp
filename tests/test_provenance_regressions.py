"""Local PDF and retrieval regressions; no model downloads or production stores."""
import asyncio
import json
from argparse import Namespace

import numpy as np
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from local_faiss_mcp import cli, server
from local_faiss_mcp.document_parser import parse_pdf_pages


class Encoder:
    def encode(self, texts, **kwargs):
        return np.array([[len(text), 1, 1] for text in texts], dtype='float32')


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'SentenceTransformer', lambda *a, **k: Encoder())
    result = server.FAISSVectorStore(str(tmp_path / 'index'), str(tmp_path / 'metadata.json'))
    monkeypatch.setattr(server, 'vector_store', result)
    return result


def make_pdf(path, texts):
    writer = PdfWriter()
    font = writer._add_object(DictionaryObject({
        NameObject('/Type'): NameObject('/Font'),
        NameObject('/Subtype'): NameObject('/Type1'),
        NameObject('/BaseFont'): NameObject('/Helvetica'),
    }))
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject('/Resources')] = DictionaryObject({
            NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 72 720 Td ({text}) Tj ET'.encode('ascii'))
        page[NameObject('/Contents')] = writer._add_object(stream)
    with path.open('wb') as output:
        writer.write(output)


@pytest.mark.parametrize('rerank', [False, True])
def test_query_returns_complete_chunk(store, rerank):
    text = 'retrieved words ' * 100 + 'UNIQUE END OF CHUNK'
    store.ingest(text, source='test')
    if rerank:
        class Reranker:
            def predict(self, pairs):
                return np.ones(len(pairs))
        store.reranker = Reranker()
    response = asyncio.run(server.call_tool('query_rag_store', {'query': 'words'}))[0].text
    assert f'   Text: {text}\n\n' in response


def test_cli_pdf_preserves_pages_and_boundaries(store, tmp_path, monkeypatch):
    pdf = tmp_path / 'two-pages.PDF'
    make_pdf(pdf, ['FIRST ' * 620, 'SECOND ' * 620])
    monkeypatch.setattr(cli, 'get_faiss_config', lambda: {
        'index_dir': str(tmp_path), 'embed_model': 'all-MiniLM-L6-v2'})
    monkeypatch.setattr(cli, 'FAISSVectorStore', lambda **kwargs: store)
    assert cli.cmd_index(Namespace(files=[str(pdf)], recursive=False)) == 0
    # Inspect persisted records too, so this checks the storage boundary.
    docs = json.loads((tmp_path / 'metadata.json').read_text())['documents']
    assert [doc['metadata']['pdf_page'] for doc in docs] == [1, 1, 2, 2]
    for doc in docs:
        page = doc['metadata']['pdf_page']
        assert doc['metadata']['local_path'] == str(pdf.resolve())
        assert doc['source'] == str(pdf)
        assert set(doc['text'].split()) == {('FIRST' if page == 1 else 'SECOND')}


def test_blank_pdf_page_does_not_renumber_physical_pages(tmp_path):
    pdf = tmp_path / 'blank-first.pdf'
    make_pdf(pdf, ['', 'page two'])
    assert parse_pdf_pages(pdf) == [{'text': 'page two', 'metadata': {'pdf_page': 2}}]


def test_metadata_merge_reload_and_legacy_query(store, monkeypatch):
    store.ingest('', metadata={'edition': '2025', 'pdf_page': 9}, pages=[
        {'text': 'page text', 'metadata': {'pdf_page': 2}}])
    loaded = server.FAISSVectorStore(store.index_path, store.metadata_path)
    assert loaded.query('page')[0]['metadata'] == {'edition': '2025', 'pdf_page': 2}
    del loaded.metadata['documents'][0]['metadata']
    monkeypatch.setattr(server, 'vector_store', loaded)
    assert 'page text' in asyncio.run(server.call_tool('query_rag_store', {'query': 'page'}))[0].text
