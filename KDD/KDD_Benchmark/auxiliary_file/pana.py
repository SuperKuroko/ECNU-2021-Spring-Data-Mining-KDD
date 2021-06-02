#encoding: utf-8
import os
import sys
import importlib
import json
importlib.reload(sys)
sys.path.append("../")
import config

if __name__ == '__main__':
    pana_file_path = os.path.join(config.DATASET_PATH,"paperIdAuthorId_to_name_and_affiliation.json")
    print("Loading...")
    pana = json.load(open(pana_file_path))
    print("Finished!")
    while True:
        try:
            pa = input()
            if pa.count("|") == 1:
                paperId,authorId = pa.split("|")
                if paperId.isdigit() and authorId.isdigit():
                    print(pana[pa])
                else:
                    break
            else:
                break
        except:
            break

    