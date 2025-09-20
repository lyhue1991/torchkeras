import sys 
class Ollama:
    def __init__(self,
                 model='qwen2',
                 max_chat_rounds=20,
                 stream=True,
                 system=None,
                 history=None
                ):
        self.model = model
        self.history = [] if history is None else history
        self.max_chat_rounds = max_chat_rounds
        self.stream = stream
        self.system = system 
        
        try:
            self.register_magic() 
            response = self('你好')
            if not self.stream:
                print(response)
            print('register magic %%chat sucessed ...',file=sys.stderr)
            self.history = self.history[:-1]
        except Exception as err:
            print('register magic %%chat failed ...',file=sys.stderr)
            print(err)
             
    @classmethod
    def build_messages(cls,query=None,history=None,system=None):
        messages = []
        history = history if history else [] 
        if system is not None:
            messages.append({'role':'system','content':system})
        for prompt,response in history:
            pair = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
            messages.extend(pair)
        if query is not None:
            messages.append({"role": "user", "content": query})
        return messages

    def chat(self, messages, stream=True):
        from openai import OpenAI
        client = OpenAI(
            base_url='http://localhost:11434/v1/',
            api_key='ollama'
        )
        completion = client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=stream
        )    
        return completion
        
        
    def __call__(self,query):
        from IPython.display import display,clear_output 
        len_his = len(self.history)
        if len_his>=self.max_chat_rounds+1:
            self.history = self.history[len_his-self.max_chat_rounds:]
        messages = self.build_messages(query=query,history=self.history,system=self.system)
        if not self.stream:
            completion = self.chat(messages,stream=False)
            response = completion.choices[0].message.content 
            self.history.append((query,response))
            return response 
        
        completion = self.chat(messages,stream=True)

        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content
            print(response)
            clear_output(wait=True)
        self.history.append((query,response))
        return response 
    
    def register_magic(self):
        import IPython
        from IPython.core.magic import (Magics, magics_class, line_magic,
                                        cell_magic, line_cell_magic)
        @magics_class
        class ChatMagics(Magics):
            def __init__(self,shell, pipe):
                super().__init__(shell)
                self.pipe = pipe

            @line_cell_magic
            def chat(self, line, cell=None):
                "Magic that works both as %chat and as %%chat"
                if cell is None:
                    return self.pipe(line)
                else:
                    print(self.pipe(cell))       
        ipython = IPython.get_ipython()
        magic = ChatMagics(ipython,self)
        ipython.register_magics(magic)
        