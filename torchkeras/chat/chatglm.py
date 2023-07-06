class ChatGLM(object):
    def __init__(self,
                 model,
                 tokenizer,
                 max_chat_rounds=20,
                 history=None,
                 max_length=8192,
                 num_beams=1,
                 do_sample=True,
                 top_p=0.8,
                 temperature=0.8,
                 logits_processor=None
                ):
        
        self.__dict__.update(locals())
        self.history =  [] if history is None else history
        
        try:
            self.register_magic() 
            print('register magic %%chatglm sucessed ...')
        except Exception as err:
            print('register magic %%chatglm failed ...')
            print(err)
        
        response = self('你好')
        print(response)
        self.history = self.history[:-1]

    
    def __call__(self,query):
        if len(self.history)>=self.max_chat_rounds+1:
            self.history = [self.history[0]]+self.history[2:]
            
        response,self.history  = self.model.chat(self.tokenizer,
            query,self.history,self.max_length,self.num_beams,
            self.do_sample,self.top_p,self.temperature,
            self.logits_processor)

        return response 
    
    def register_magic(self):
        import IPython
        from IPython.core.magic import (Magics, magics_class, line_magic,
                                        cell_magic, line_cell_magic)
        @magics_class
        class ChatGLMMagics(Magics):
            def __init__(self,shell, pipe):
                super().__init__(shell)
                self.pipe = pipe

            @line_cell_magic
            def chatglm(self, line, cell=None):
                "Magic that works both as %chatglm and as %%chatglm"
                if cell is None:
                    return self.pipe(line)
                else:
                    print(self.pipe(cell))
                    
        ipython = IPython.get_ipython()
        magic = ChatGLMMagics(ipython,self)
        ipython.register_magics(magic)
        