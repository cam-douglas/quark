from utilities.doc_utils import open_doc

def test_docs_index_exists():
    text = open_doc("INDEX.md")
    assert text.strip().startswith("# Documentation Index")
