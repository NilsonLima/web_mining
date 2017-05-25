import nltk
import json
import os

from nltk.corpus import stopwords

macro = "docbase/original/macroeconomics/"
bio = "docbase/original/political_leader_biographies/"

def stemming(title, content):
    ps = nltk.stem.PorterStemmer( )

    title = [ps.stem(i) for i in title.split( )]
    content = [ps.stem(i) for i in content.split( )]

    title = write_list_as_txt(title)
    content = write_list_as_txt(content)

    return title, content

def stoplist(title, content):
    stops = stopwords.words('english')

    title = list(filter(lambda x: x not in stops, title.split( )))
    content = list(filter(lambda x: x not in stops, content.split( )))

    title = write_list_as_txt(title)
    content = write_list_as_txt(content)

    return title, content

def write_list_as_txt(txt):
    str = ""

    for i in range(len(txt)):
        str += txt[i] + " "

    return str

def main( ):
    path = "docbase/stem_stop/political_leader_biographies/"
    length = len(os.listdir(bio))

    stem_flag = True
    stop_flag = True

    for i in range(length):
        with open(bio + str(i) + ".json", "r") as f:
            w = json.load(f)

        title = w["title"]
        content = w["content"]

        if(stop_flag):
            title, content = stoplist(title, content)

        if(stem_flag):
            title, content = stemming(title, content)

        w = {"title": title, "content": content}

        with open(path + str(i) +".json", "w") as f:
            json.dump(w, f)

    return

if __name__ == '__main__':
    main( )
