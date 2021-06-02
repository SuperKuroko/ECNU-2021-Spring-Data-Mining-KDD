#encoding: utf-8
import os
import sys
import importlib
import json
importlib.reload(sys)
sys.path.append("../")
import config

if __name__ == '__main__':
    coauthor_file_path = os.path.join(config.DATASET_PATH,"coauthor.json")
    coauthor = json.load(open(coauthor_file_path))
    while True:
        try:
            authorId = input()
            if authorId.isdigit():
                print(coauthor[authorId])
            else:
                break
        except:
            break

    