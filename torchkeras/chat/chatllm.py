import torch
import sys 
from copy import deepcopy
from .conversations import conv_templates, get_conv_template 
from .text2ids import build_inputs_labels

#chat tool for chatglm2-6b,baichuan-13b,internlm-chat-7b,qwen-7b-chat and more...
class ChatLLM:
    def __init__(self,model,tokenizer,
                 model_type=None,
                 max_chat_rounds=20,
                 max_new_tokens=512,
                 stream=True,
                 history=None,
                 stop_words_ids=None
                ):
        self.model = model
        self.tokenizer = tokenizer
        
        if not self.tokenizer.eos_token_id:
            self.tokenizer.eos_token_id =  (model.config.eos_token_id 
                or model.generation_config.eos_token_id)
            
        self.model_type = model_type if model_type else self.get_model_type() 
        conv = get_conv_template(self.model_type)
        self.conv_template = conv

        stop_words_ids = stop_words_ids if stop_words_ids else []
        stop_token_ids = [[w] for w in conv.stop_token_ids] if conv.stop_token_ids else []
        self.stop_words_ids = stop_token_ids + stop_words_ids
        
        self.model.generation_config.stop_words_ids = self.stop_words_ids
        self.model.generation_config.max_new_tokens = max_new_tokens
        self.model.eval()
        self.history = [] if history is None else history
        self.max_chat_rounds = max_chat_rounds
        self.stream = stream
        
        try:
            self.register_magic() 
            response = self('你好')
            if not self.stream:
                print(response)
            print('register magic %%chat sucessed ...',file=sys.stderr)
            self.history = self.history[:-1]
        except Exception as err:
            print('register magic %%chat failed ...',file=sys.stderr)
            raise err 
        
    def get_model_type(self):
        model_cls = str(self.model.__class__).split('.')[-1].lower()[:-2] 
        keys = list(conv_templates.keys()) 
        max_same,most_type = 0,None
        for k in keys:
            same = 0
            for a,b in zip(k,model_cls):
                if a==b:
                    same+=1
                else:
                    break
            if same>max_same:
                max_same = same
                most_type = k 
        if max_same>=3:
            return most_type
        else:
            raise Exception('Error: get_model_type failed @ model_cls='+model_cls)
            return None
        
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

    def build_conversations(self,messages):
        conv = deepcopy(self.conv_template)
        msgs_sys = [d for d in messages if d['role']=='system']
        if msgs_sys:
            conv.set_system_message(msgs_sys[0]['content'])
        for d in messages:
            if d['role']=='user':
                conv.append_message(conv.roles[0], d['content'])
            elif d['role']=='assistant':
                conv.append_message(conv.roles[1], d['content'])
            else:
                raise Exception('role must be one of (system,user,assistant)')
        if d['role']!='assistant':
            conv.append_message(conv.roles[1], None)
        return conv
        
    def build_prompt(self,messages):
        conv = self.build_conversations(messages)
        return conv.get_prompt()

    def build_inputs_labels(self,messages,multi_rounds=True):
        conv = self.build_conversations(messages)
        inputs,labels = build_inputs_labels(
            conv, self.tokenizer, multi_rounds=multi_rounds)
        return inputs,labels
        
    def chat(self, messages, stream=False, generation_config=None):
        model,tokenizer = self.model,self.tokenizer
        #prompt = self.build_prompt(messages)
        #inputs = tokenizer([prompt],add_special_tokens=False)
        input_ids,labels = self.build_inputs_labels(messages)
        inputs = {'input_ids': torch.tensor([input_ids]).to(model.device)}
        if generation_config is not None:
            generation_config = deepcopy(model.generation_config.update(**generation_config))
        else:
            generation_config =  deepcopy(model.generation_config)

        stop_words_ids = self.stop_words_ids

        if not stream:
            from transformers import PreTrainedModel
            model.__class__.generate = PreTrainedModel.generate  # disable stream
            output_ids = model.generate(
                 **inputs,
                 stop_words_ids = stop_words_ids,
                 generation_config = generation_config,
                 return_dict_in_generate = False,
            )
            output_ids = output_ids[0][len(inputs["input_ids"][0]):]
            end_token_idx = 0
            should_stop = False
            for end_token_idx in range(len(output_ids)):
                for stop_ids in stop_words_ids:
                    if output_ids[end_token_idx-len(stop_ids):end_token_idx].tolist()==stop_ids:
                        should_stop = True
                        break
                if should_stop:
                    break      
            outputs = tokenizer.decode(
                output_ids[:end_token_idx], skip_special_tokens=True
            )
            return outputs
        else:
            from .stream_generate  import NewGenerationMixin, StreamGenerationConfig
            model.__class__.generate = NewGenerationMixin.generate
            model.__class__.sample_stream = NewGenerationMixin.sample_stream
            config_dic = generation_config.to_dict()
            config_dic.update({'do_stream':True})
            stream_config = StreamGenerationConfig(**config_dic)
            
            def stream_generator():
                outputs = []
                for token in model.generate(**inputs,
                                            generation_config=stream_config,
                                            do_sample=True,
                                            stop_words_ids=stop_words_ids,
                                            return_dict_in_generate = False,
                                           ):
                    token_idx = token.item()
                    outputs.append(token_idx)
                    should_stop = False
                    for stop_ids in stop_words_ids:
                        if outputs[-len(stop_ids):]==stop_ids:
                            should_stop = True
                            break
                    if should_stop:
                        break
                    yield tokenizer.decode(outputs, skip_special_tokens=True)
            return stream_generator()
        
    def __call__(self,query):
        from IPython.display import display,clear_output 
        len_his = len(self.history)
        if len_his>=self.max_chat_rounds+1:
            self.history = self.history[len_his-self.max_chat_rounds:]
        messages = self.build_messages(query=query,history=self.history)
        if not self.stream:
            response = self.chat(messages,stream=False)
            self.history.append((query,response))
            return response 
        
        result = self.chat(messages,stream=True)
        for response in result:
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
        