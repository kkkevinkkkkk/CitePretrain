# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pymongo import MongoClient
import requests
import os
import urllib.request
from bs4 import BeautifulSoup
import urllib.parse as urlparse
from urllib.parse import parse_qs
import itertools

DEFAULT_MONGO_CONNECTION_STRING = os.getenv(
    "MONGO_CONNECTION_STRING", "mongodb://127.0.0.1:27017/admin"
)



class KnowledgeSource:
    def __init__(
        self,
        mongo_connection_string=None,
        database="cite_pretrain",
        collections=None,
    ):
        if not mongo_connection_string:
            mongo_connection_string = DEFAULT_MONGO_CONNECTION_STRING
        self.client = MongoClient(mongo_connection_string)
        self.db = self.client[database]
        self.collections = self.db.list_collection_names() if collections is None else collections
        collections_size = [
            (collection, self.db[collection].estimated_document_count()) for collection in self.collections
        ]
        collections_size.sort(key=lambda x: x[1])
        # print("Collections sorted by size:")
        # for collection, size in collections_size:
        #     print(f"{collection}: {size}")
        self.collections = [collection for collection, _ in collections_size]

    def get_all_pages_cursor(self):
        cursor = itertools.chain.from_iterable(
            self.db[collection].find({}) for collection in self.collections
        )
        return cursor

    def get_num_pages(self):
        # Return the total number of documents across all collections
        total_num = 0
        for collection in self.collections:
            total_num += self.db[collection].estimated_document_count()
        return total_num

    def get_page_by_id(self, doc_id):
        for collection in self.collections:
            page = self.db[collection].find_one({"_id": str(doc_id)})
            if page:
                return page
        return None


    def get_page_by_title(self, title, attempt=0):
        for collection in self.collections:
            page = self.db[collection].find_one({"title": str(title)})
            if page:
                return page
        return None


def _get_pageid_from_api(title, client=None):
    pageid = None

    title_html = title.strip().replace(" ", "%20")
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&titles={}&format=json".format(
            title_html
        )
    )

    try:
        # Package the request, send the request and catch the response: r
        r = requests.get(url)

        # Decode the JSON data into a dictionary: json_data
        json_data = r.json()

        if len(json_data["query"]["pages"]) > 1:
            print("WARNING: more than one result returned from wikipedia api")

        for _, v in json_data["query"]["pages"].items():
            pageid = v["pageid"]

    except Exception as e:
        #  print("Exception: {}".format(e))
        pass

    return pageid


def _read_url(url):
    with urllib.request.urlopen(url) as response:
        html = response.read()
        soup = BeautifulSoup(html, features="html.parser")
        title = soup.title.string.replace(" - Wikipedia", "").strip()
    return title


def _get_title_from_wikipedia_url(url, client=None):
    title = None
    try:
        title = _read_url(url)
    except Exception:
        try:
            # try adding https
            title = _read_url("https://" + url)
        except Exception:
            #  print("Exception: {}".format(e))
            pass
    return title


class WikiKnowledgeSource:
    def __init__(
        self,
        mongo_connection_string=None,
        database="kilt",
        collection="knowledgesource",
    ):
        if not mongo_connection_string:
            mongo_connection_string = DEFAULT_MONGO_CONNECTION_STRING
        self.client = MongoClient(mongo_connection_string)
        self.db = self.client[database][collection]

    def get_all_pages_cursor(self):
        cursor = self.db.find({})
        return cursor

    def get_num_pages(self):
        return self.db.estimated_document_count()

    def get_page_by_id(self, wikipedia_id):
        page = self.db.find_one({"_id": str(wikipedia_id)})
        # page = self.db.find_one({"_id": int(wikipedia_id)})
        return page

    def get_page_by_title(self, wikipedia_title, attempt=0):
        page = self.db.find_one({"wikipedia_title": str(wikipedia_title)})
        return page

    def get_page_from_url(self, url):
        page = None

        # 1. try to look for title in the url
        parsed = urlparse.urlparse(url)
        record = parse_qs(parsed.query)
        if "title" in record:
            title = record["title"][0].replace("_", " ")
            page = self.get_page_by_title(title)

        # 2. try another way to look for title in the url
        if page == None:
            title = url.split("/")[-1].replace("_", " ")
            page = self.get_page_by_title(title)

        # 3. try to retrieve the current wikipedia_id from the url
        if page == None:
            title = _get_title_from_wikipedia_url(url, client=self.client)
            if title:
                pageid = _get_pageid_from_api(title, client=self.client)
                if pageid:
                    page = self.get_page_by_id(pageid)

        return page
