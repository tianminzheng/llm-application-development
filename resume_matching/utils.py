import re
import pandas as pd
from llama_index.core.schema import TransformComponent
from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document

class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = re.sub(r"[^0-9A-Za-z ]", "", node.text)
            node.text = node.text.lower()
        return nodes
    
class MyFileReader(BaseReader):
    def load_data(self, file, extra_info=None):
        text = ''
        df = pd.read_csv(file)
        text_list = df.apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
        # load_data returns a list of Document objects
        return [Document(text=text + "Foobar", extra_info=extra_info or {'file_name': i}) \
                 for i, text in enumerate(text_list)]

def clean_text(text):
    text = re.sub(r"[^0-9A-Za-z ]", "", text)
    return text.lower()