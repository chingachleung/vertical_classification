import re
import csv
import pycld2 as cld2
from bs4 import BeautifulSoup
import lxml
import os
import regex

# function to remove bad characters that cause Utf-8 encoding error
RE_BAD_CHARS = regex.compile(r"[\p{Cc}\p{Cs}]+")


def remove_bad_chars(text):
    return RE_BAD_CHARS.sub("", text)


def data_extraction():
    """
    data extraction method from xml files in a folder to write into csv file, using Fred's suggestion
    :param
    """
    directory = 'test_data/vertical_files'

    with open("challenge_set.csv", 'w', encoding='utf-8-sig') as nfile:
        field_names = ['sources', 'categories', 'subcategories', "targetids"]
        writer = csv.DictWriter(nfile, fieldnames=field_names,
                                delimiter=",",
                                extrasaction="ignore",
                                escapechar="\\",
                                lineterminator="\n", )
        writer.writeheader()
        for filename in os.listdir(directory):
            category = re.findall(r"#category#(.+?)#", filename)
            subcategory = re.findall(r"#subcategory#(.+?)#", filename)
            # why not use digits
            targetid = re.findall(r"#targetid#(\d+)", filename)
            if len(category) > 0:
                category = category[0]
                if category == 'medicaldevices':
                    category = 'medicaldevice'
                elif category == 'lifesciences-PFA':
                    category = 'lspfa'
            else:
                continue
            if len(subcategory) > 0:
                subcategory = subcategory[0]
            else:
                subcategory = 'unknown'
            if len(targetid) > 0:
                targetid = targetid[0]
            else:
                targetid = 'unknown'
            curr_scr = []
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                with open(f, "rb") as infile:
                    soup = BeautifulSoup(infile, parser='lxml')
                    trans = soup.find_all('trans-unit')
                    for t in trans:
                        if len(t.find_all('alt-trans')) > 0:
                            scr = t.source.text
                            s_soup = BeautifulSoup(scr, features='html.parser')
                            scr = s_soup.get_text().strip()
                            scr = re.sub("\s{2,}", " ", scr)
                            scr = re.sub('<.*?>', "", scr)
                            scr = re.sub('\\\\r\\\\n', '', scr)
                            scr = scr.encode("ascii", "ignore")
                            scr = scr.decode()
                            scr = remove_bad_chars(scr)
                            _, _, details = cld2.detect(scr)
                            if details[0][0] == "ENGLISH":
                                # remove empty strings
                                if len(scr) > 0:
                                    curr_scr.append(scr)

                    joined_scr = " ".join(curr_scr)
                # remove empty doc
                if len(joined_scr) > 0:
                    writer.writerow({"sources": joined_scr, "categories": category, "subcategories": subcategory,
                                     "targetids": targetid})