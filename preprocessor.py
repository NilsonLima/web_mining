import json
import re
import os

macro = "docbase/original/macroeconomics/"
bio = "docbase/original/political_leader_biographies/"

def process(path):
    length = len(os.listdir(path))

    for i in range(length):
        with open(path + str(i) + ".json", "r") as w:
            f = json.load(w)

        content = re.sub("<[^>]*>", "", f["content"])
        content = re.sub(r'([^\s\w]|_)+', '', content)
        title = re.sub("<[^>]*>", "", f["title"])
        title = re.sub(r'([^\s\w]|_)+', '', title)

        f = {"title": title, "content": content}

        with open(path + str(i) +".json", "w") as w:
            json.dump(f, w)

    return

def main( ):
    process(macro)
    process(bio)

    return

if __name__ == '__main__':
    main( )
