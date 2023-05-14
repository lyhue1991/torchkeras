class ChatGPT(object):
    def __init__(self,
                 api_key = None,
                 max_chat_rounds=3,
                 proxy = False,
                 model="gpt-3.5-turbo",
                 temperature=0,
                 system_content='You are friendly chatbot.'
                ):
        
        import json
        from pathlib import Path 
        json_file = Path(__file__).parent/'openai.json'
        if api_key is not None:
            with open(json_file,'w') as fp:
                json.dump({'api_key':api_key},fp)
        else:
            with open(json_file,'r') as fp:
                api_key = json.load(fp)['api_key']
        
        self.api_key = api_key 
        self.model = model
        self.max_chat_rounds = max_chat_rounds
        self.temperature = temperature
        self.messages =  [{'role':'system', 'content':system_content}]
        
        if  proxy==False:
            import openai
            openai.api_key  = api_key
            test_prompt = [{'role':'user','content':'hello'}]
            reply = self.get_openai_completion(self.messages+test_prompt)
            self.get_completion_from_messages = self.get_openai_completion
            print(reply)
            
        else:
            url = "https://closeai.deno.dev/v1/chat/completions"
            #url = "https://chatai.1rmb.tk/api/v1/chat/completions"
            #url =  "https://api.lixining.com/v1/chat/completions"  
            self.proxy = url if not isinstance(proxy,str) else proxy 
            test_prompt = [{'role':'user','content':'你好'}]
            reply = self.get_proxy_completion(self.messages+test_prompt)
            self.get_completion_from_messages = self.get_proxy_completion
            print(reply)
        try:
            self.register_magic() 
            print('register magic %%chatgpt sucessed ...')
        except Exception as err:
            print('register magic %%chatgpt failed ...')
            print(err)
            
    def get_openai_completion(self,messages):
        import openai 
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature, # this is the degree of randomness of the model's output
        )
        return response.choices[0].message["content"]

    def get_proxy_completion(self,messages):  
        
        import requests 
        headers = {
          'Authorization': f'Bearer {self.api_key}',
          'Content-Type': 'application/json'
        }

        payload = {
          "model": self.model,
          "messages": messages
        }
        
        try:
            response = requests.post(self.proxy, headers=headers, json=payload)
            response.raise_for_status() 
            data = response.json()
            result = (data["choices"][0]["message"]["content"])
            return result
        except Exception as err:
            print(f"error: {err}")
            return err
    
    def __call__(self,prompt):
        if len(self.messages)>=2*self.max_chat_rounds+1:
            self.messages = [self.messages[0]]+self.messages[3:]
        self.messages.append({'role':'user','content':prompt})
        reply = self.get_completion_from_messages(messages = self.messages)
        self.messages.append({'role':'assistant','content':reply})
        return reply  
    
    def register_magic(self):
        import IPython
        from IPython.core.magic import (Magics, magics_class, line_magic,
                                        cell_magic, line_cell_magic)
        @magics_class
        class ChatGPTMagics(Magics):
            def __init__(self,shell, pipe):
                super().__init__(shell)
                self.pipe = pipe

            @line_cell_magic
            def chatgpt(self, line, cell=None):
                "Magic that works both as %chatgpt and as %%chatgpt"
                if cell is None:
                    return self.pipe(line)
                else:
                    print(self.pipe(cell))
                    
        ipython = IPython.get_ipython()
        magic = ChatGPTMagics(ipython,self)
        ipython.register_magics(magic)