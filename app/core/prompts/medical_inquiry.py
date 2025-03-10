import os
# from typing import List, Tuple

class LivePromptObject(str):
    def __init__(self, prompt_file_path: str, read_as_eval: bool = False):
        self.prompt_file_path = prompt_file_path
        self.read_as_eval = read_as_eval
        
    @property
    def value(self):
        return self()
    
    def __call__(self, *args, **kwargs):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, self.prompt_file_path)
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        # If this is the summary few shot file, evaluate it as Python code
        if self.read_as_eval:
            return eval(content)
            
        # If kwargs are provided, format the content with them
        if kwargs:
            return content.format(**kwargs)
            
        return str(content)

    def __str__(self):
        return str(self())

    def __len__(self):
        return len(self.__str__())
        
    def format(self, *args, **kwargs):
        content = self()
        if isinstance(content, str):
            return content.format(*args, **kwargs)
        return content

    # def __getattr__(self, name):
    #     return self()
    
    def __getitem__(self, key):
        return self.value[key]


SYSTEM_PROMPT:str = LivePromptObject("medical_inquiry/system_prompt.txt")

ENTITY_PROMPT_KO:str = LivePromptObject("medical_inquiry/entity_prompt_ko.txt")

ENTITY_PROMPT_EN:str = LivePromptObject("medical_inquiry/entity_prompt_en.txt")

TIMER_PROMPT:str = LivePromptObject("medical_inquiry/timer_prompt.txt")

MULTI_QUERY_PROMPT:str = LivePromptObject("medical_inquiry/multi_query_prompt.txt")
