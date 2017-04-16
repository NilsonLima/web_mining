import sys
import os
import json
import argparse
from elasticsearch import Elasticsearch

macro = "docbase/original/macroeconomics/"
bio = "docbase/original/political_leader_biographies/"

def index_type( ):
    stem = False
    stop = False

    if("stem" in FLAGS.index.split("_")):
        stem = True

    if("stop" in FLAGS.index.split("_")):
        stop = True

    return stop, stem


def create_settings(stop, stem):
    fltr = ["lowercase"]

    if(stem):
        fltr.append("my_stem")

    if(stop):
        fltr.append("my_stopwords")

    settings = {
        "settings": {
            "analysis": {
                "filter": {
                    "my_stopwords": {
                        "type": "stop",
                        "stopwords": "_english_"
                    },
                    "my_stem": {
                        "type": "stemmer",
                        "name": "english"
                    }
                },
                "analyzer": {
                    "my_search_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "my_stopwords"]
                    },
                    "my_analyzer": {
                        "type": "custom",
                        "char_filter": ["html_strip"],
                        "tokenizer": "standard",
                        "filter": fltr
                    }
                }
            }
        },
        "mappings": {
            "books": {
                "properties":{
                    "content": {
                        "type": "text",
                        "analyzer": "my_analyzer"
                        # "search_analyzer": "my_search_analyzer"
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "my_analyzer"
                        # "search_analyzer": "my_search_analyzer"
                    }
                }
            }
        }
    }

    return settings


def bulk(es):
    num_docs = 100

    doc = {}
    for i in range(num_docs):
        with open(macro + str(i) + ".json", "r") as w:
            f = json.load(w)

        doc["title"] = f["title"]
        doc["content"] = f["content"]

        es.index(index = FLAGS.index, doc_type = "books", id = i, body = doc)

    doc = {}
    for i in range(num_docs, 2 * num_docs):
        with open(bio + str(i - num_docs) + ".json", "r") as w:
            f = json.load(w)

        doc["title"] = f["title"]
        doc["content"] = f["content"]

        es.index(index = FLAGS.index, doc_type = "books", id = i, body = doc)

    print("refreshing...")
    es.indices.refresh(index = FLAGS.index)

    return


def main( ):
    es = Elasticsearch( )

    if not es.indices.exists(FLAGS.index):
        stop, stem = index_type( )
        settings = create_settings(stop, stem)

        print("creating '%s' index..." % FLAGS.index)
        res = es.indices.create(index = FLAGS.index, body = settings)

        bulk(es)

    res = es.search(index = FLAGS.index, size = 200, body = {"query": { \
                                                                "query_string": { \
                                                                    "fields": ["content", "title"], \
                                                                    # "analyzer": q_analyzer,\
                                                                    "query": FLAGS.query \
                                                                    }
                                                                }
                                                            })

    print("total hits %d" % res['hits']['total'])
    print("documents returned:")

    for hit in res['hits']['hits']:
        print(hit["_id"])

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser( )
    parser.add_argument('--query', type=str, default="", help='Defines a query.')
    parser.add_argument('--index', type=str, default="original", help='Index name.')

    FLAGS, unparsed = parser.parse_known_args( )
    main( )
