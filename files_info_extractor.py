"""script return language and the number of word from a document
"""

import argparse
import json
from file_reader.file_reader import FilesReader


class FilesInfoExtractor(object):
    """Extract information form files as:
       number of words and lang
    """

    def __init__(self):
        self.file_reader = FilesReader()

    def extract_info_from_file(self, filename):
        return self.file_reader.extract_infos_from_file(filename)

    @staticmethod
    def save_json_file(data, output_name):
        with open(output_name + ".json", 'w') as outfile:
            json.dump(data, outfile)

    def run(self, filename, output_name):
        data = self.extract_info_from_file(filename)
        FilesInfoExtractor.save_json_file(data, output_name)

def arguments():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument(
        'filename', help='filename.')
    parser.add_argument(
        'output_name', help='json file where we put the final result.')
    return parser.parse_args()


def main(args):
    FilesInfoExtractor().run(args.filename, args.output_name)


if __name__ == '__main__':
    args = arguments()
    main(args)
