import sys 
class ChatLLM:
    def __init__(self,model,tokenizer,
                 model_type=None,
                 max_chat_rounds=20,
                 max_new_tokens=512,
                 stream=True,
                 history=None
                ):
        self.model = model
        self.tokenizer = tokenizer
        self.model.generation_config.max_new_tokens = max_new_tokens
        self.model.eval()
        self.history = [] if history is None else history
        self.max_chat_rounds = max_chat_rounds
        self.stream = stream
        
        try:
            self.register_magic() 
            response = self('你好')
            print(response)
            print('register magic %%chat sucessed ...',file=sys.stderr)
            self.history = self.history[:-1]
        except Exception as err:
            print('register magic %%chat failed ...',file=sys.stderr)
            print(err, file=sys.stderr)
            
            
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
        
        model,tokenizer = self.model,self.tokenizer
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        if stream:
            from transformers import TextStreamer
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        else:
            streamer = None

        generated_ids = model.generate(
            **model_inputs,
            streamer  = streamer 
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, 
            output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response 
        

    def __call__(self,query):
        
        len_his = len(self.history)
        if len_his>=self.max_chat_rounds+1:
            self.history = self.history[len_his-self.max_chat_rounds:]
        messages = self.build_messages(query=query,history=self.history)
        response = self.chat(messages,stream=self.stream)
        self.history.append((query,response))
        
        if self.stream:
            from IPython.display import clear_output,display
            clear_output(wait=True) 
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
        