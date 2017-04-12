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


def create_settings(stopwords, stemming, stop, stem):
    fltr = ["lowercase"]

    if(stem):
        fltr.append("my_stem")

    if(stop):
        fltr.append("my_stopwords")

    settings = {
        "mappings": {
            "books": {
                "properties":{
                    "content": {
                        "type": "text",
                        "analyzer": "my_analyzer",
                        "search_analyzer": "my_analyzer"
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "my_analyzer",
                        "search_analyzer": "my_analyzer"
                    }
                }
            }
        },
        "settings": {
            "analysis": {
                "filter": {
                    "my_stopwords": stopwords,
                    "my_stem": stemming
                },
                "analyzer": {
                    "my_analyzer": {
                        "type": "custom",
                        "char_filter": ["html_strip"],
                        "tokenizer": "standard",
                        "filter": fltr
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

    stemming = {}
    stopwords = {}

    if not es.indices.exists(FLAGS.index):
        stop, stem = index_type( )

        if(stop):
            stopwords["type"] = "stop"
            stopwords["stopwords"] = "_english_"

        if(stem):
            stemming["type"] = "snowball"
            stemming["language"] = "English"

        settings = create_settings(stopwords, stemming, stop, stem)

        print("creating '%s' index..." % FLAGS.index)
        res = es.indices.create(index = FLAGS.index, body = settings)
        print("response: '%s'" % res)

        print("bulk indexing...")
        bulk(es)

    res = es.search(index = FLAGS.index, size = 100, body = {"query": {"query_string": {"fields": ["content", "title"], "query": FLAGS.query}}})

    #print(res['hits']['hits'])
    #print("response: '%s'" % res)

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
