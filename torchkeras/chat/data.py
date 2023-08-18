from .conversations import SeparatorStyle 

def data_collator(examples: list):
    len_ids = [len(example["input_ids"]) for example in examples]
    longest = max(len_ids) 
    
    input_ids = []
    labels_list = []
    
    for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
        ids = example["input_ids"]
        labs = example["labels"]
        
        ids = ids + [tokenizer.pad_token_id] * (longest - length)
        labs = labs + [-100] * (longest - length)
        
        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labs))
          
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }
    
def build_inputs_labels(self, tokenizer, multi_rounds=True):
    system_prompt = self.system_template.format(system_message=self.system_message)
    encode_fn = lambda text: tokenizer.encode(text,add_special_tokens=False)
    ignore_fn = lambda arr:[-100 for _ in arr]
    eos = [tokenizer.eos_token_id]
    
    if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
        inputs = encode_fn(system_prompt) + encode_fn(self.sep)
        labels = ignore_fn(inputs)
        for i,(role, message) in enumerate(self.messages):
            if message:
                pre,msg,post = [encode_fn(x) for x in [role + ": ",message,self.sep]]
                inputs += (pre + msg + eos + post)
                if role==self.roles[0] or (multi_rounds and i<len(self.messages)-1):
                    labels+= ignore_fn(pre + msg + eos + post)
                else:
                    labels+= (ignore_fn(pre) + msg + eos + ignore_fn(post))
            else:
                pre = encode_fn(role + ":")
                inputs += pre
                labels += ignore_fn(pre)
        return inputs,labels
        
    elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
        seps = [self.sep, self.sep2]
        inputs = encode_fn(system_prompt) + eoncde_fn(seps[0])
        labels = ignore_fn(inputs)
        for i, (role, message) in enumerate(self.messages):
            if message:
                pre,msg,post = [encode_fn(x) 
                                for x in [role + ": ",message,seps[i % 2]]]
                inputs += (pre + msg + eos + post)
                if role==self.roles[0] or (multi_rounds and i<len(self.messages)-1):
                    labels+= ignore_fn(pre + msg + eos + post)
                else:
                    labels+= (ignore_fn(pre) + msg + eos + ignore_fn(post))
            else:
                pre = encode_fn(role + ":")
                inputs += pre
                labels += ignore_fn(pre)
        return inputs,labels
        
    elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
        inputs = encode_fn(system_prompt) + eoncde_fn(self.sep)
        labels = ignore_fn(inputs)
        for role, message in self.messages:
            if message:
                pre,msg,post = [encode_fn(x) 
                                for x in [role + ": ",message,self.sep]]
                inputs += (pre + msg + eos + post)
                if role==self.roles[0] or (multi_rounds and i<len(self.messages)-1):
                    labels+= ignore_fn(pre + msg + eos + post)
                else:
                    labels+= (ignore_fn(pre) + msg + eos + ignore_fn(post))
                    
            else:
                pre = encode_fn(role + ": ")
                inputs += pre
                labels += ignore_fn(pre)
        return inputs,labels
        
    elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:        
        inputs = [] if system_prompt == "" else encode_fn(
            system_prompt) + eoncde_fn(self.sep)
        labels = ignore_fn(inputs)
        
        for role, message in self.messages:
            if message:
                pre,msg,post = [encode_fn(x) 
                                for x in [role + "\n",message,self.sep]]
                inputs += (pre + msg + eos + post)
                if role==self.roles[0] or (multi_rounds and i<len(self.messages)-1):
                    labels+= ignore_fn(pre + msg + eos + post)
                else:
                    labels+= (ignore_fn(pre) + msg + eos + ignore_fn(post))
    
            else:
                pre = encode_fn(role + "\n")
                inputs += pre
                labels += ignore_fn(pre)
                
        return inputs,labels
        
    elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:        
        inputs = encode_fn(system_prompt)
        labels = ignore_fn(inputs)
        
        for role, message in self.messages:
            if message:
                pre,msg,post = [encode_fn(x) 
                                for x in [role,message,self.sep]]
                inputs += (pre + msg + eos + post)
                if role==self.roles[0] or (multi_rounds and i<len(self.messages)-1):
                    labels+= ignore_fn(pre + msg + eos + post)
                else:
                    labels+= (ignore_fn(pre) + msg + eos + ignore_fn(post))
            else:
                pre = encode_fn(role)
                inputs += pre
                labels += ignore_fn(pre)
        return inputs,labels
        
    elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
        seps = [self.sep, self.sep2]
        inputs = encode_fn(system_prompt)
        labels = ignore_fn(inputs)
        for i, (role, message) in enumerate(self.messages):
            if message:
                pre,msg,post = [encode_fn(x) 
                                for x in [role,message,seps[i % 2]]]
                inputs += (pre + msg + eos + post)
                if role==self.roles[0] or (multi_rounds and i<len(self.messages)-1):
                    labels+= ignore_fn(pre + msg + eos + post)
                else:
                    labels+= (ignore_fn(pre) + msg + eos + ignore_fn(post))
            else:
                pre = encode_fn(role)
                inputs += pre
                labels += ignore_fn(pre)
        return inputs,labels

    elif self.sep_style == SeparatorStyle.RWKV:
        inputs = encode_fn(system_prompt)
        labels = ignore_fn(inputs)
        for i, (role, message) in enumerate(self.messages):
            if message:
                message = message.replace("\r\n", "\n").replace("\n\n", "\n")
                pre,msg,post = [encode_fn(x) for x in [role+': ',message,"\n\n"]]
                inputs += (pre + msg + eos + post)
                if role==self.roles[0] or (multi_rounds and i<len(self.messages)-1):
                    labels+= ignore_fn(pre + msg + eos + post)
                else:
                    labels+= (ignore_fn(pre) + msg + eos + ignore_fn(post))
                    
            else:
                pre = encode_fn(role + ":")
                inputs += pre
                labels += ignore_fn(pre)
        return inputs,labels

    elif self.sep_style == SeparatorStyle.LLAMA2:
        seps = [self.sep, self.sep2]
        inputs,labels = [],[]
        for i, (role, message) in enumerate(self.messages):
            if message:
                if i == 0:
                    tokens = encode_fn(system_prompt + message + " ")
                    inputs += tokens
                    labels += ignore_fn(tokens)
                else:
                    pre,msg,post = [encode_fn(x) 
                                for x in [role + " ",message,seps[i % 2]]]
                    inputs += (pre + msg + eos + post)
                    if role==self.roles[0] or (multi_rounds and i<len(self.messages)-1):
                        labels+= ignore_fn(pre + msg + eos + post)
                    else:
                        labels+= (ignore_fn(pre) + msg + eos + ignore_fn(post))
            else:
                pre = encode_fn(role)
                inputs += pre
                labels += ignore_fn(pre)
        return inputs, labels
        
    elif self.sep_style == SeparatorStyle.CHATGLM:
        round_add_n = 1 if self.name == "chatglm2" else 0
        inputs,labels = [],[]
        if system_prompt:
            inputs += encode_fn(system_prompt + self.sep)
            labels += ignore_fn(inputs)
        for i, (role, message) in enumerate(self.messages):
            if i % 2 == 0:
                tokens = encode_fn(f"[Round {i//2 + round_add_n}]{self.sep}")
                inputs += tokens
                labels += ignore_fn(tokens)
                
            if message:
                pre,msg,post = [encode_fn(x) 
                                for x in [f"{role}：",message,self.sep]]
                inputs += (pre + msg + eos + post)
                
                if role==self.roles[0] or (multi_rounds and i<len(self.messages)-1):
                    labels+= ignore_fn(pre + msg + eos + post)
                else:
                    labels+= (ignore_fn(pre) + msg + eos + ignore_fn(post))
            else:
                pre = encode_fn(f"{role}：")
                inputs += pre
                labels += ignore_fn(pre)
        return inputs,labels

    
    elif self.sep_style == SeparatorStyle.CHATML:
        inputs,labels = [],[]
        if system_prompt:
            inputs += encode_fn(system_prompt + self.sep + "\n")
            labels += ignore_fn(inputs)
            
        for role, message in self.messages:
            if message:
                pre,msg,post = [encode_fn(x) 
                                for x in [role + "\n", message, self.sep + "\n"]]
                inputs += (pre + msg + eos + post)
                
                if role==self.roles[0] or (multi_rounds and i<len(self.messages)-1):
                    labels+= ignore_fn(pre + msg + eos + post)
                else:
                    labels+= (ignore_fn(pre) + msg + eos + ignore_fn(post))
            else:
                pre = encode_fn(role + "\n")
                inputs += pre
                labels += ignore_fn(pre)
        return inputs,labels

    
    elif self.sep_style == SeparatorStyle.CHATINTERN:
        seps = [self.sep, self.sep2]
        inputs = encode_fn(system_prompt)
        labels = ignore_fn(inputs)
        
        for i, (role, message) in enumerate(self.messages):
            if i % 2 == 0:
                tokens = encode_fn("<s>")
                inputs += tokens
                labels += ignore_fn(tokens)
                
            if message:
                pre,msg,post = [encode_fn(x) 
                                for x in [role + ":", message, seps[i % 2] + "\n"]]
                inputs += (pre + msg + eos + post)
                
                if role==self.roles[0] or (multi_rounds and i<len(self.messages)-1):
                    labels+= ignore_fn(pre + msg + eos + post)
                else:
                    labels+= (ignore_fn(pre) + msg + eos + ignore_fn(post))
            else:
                pre = encode_fn(role + ":")
                inputs += pre
                labels += ignore_fn(pre)
                
        return inputs,labels

    
    elif self.sep_style == SeparatorStyle.DOLLY:
        seps = [self.sep, self.sep2]
        inputs = encode_fn(system_prompt)
        labels = ignore_fn(inputs)
        for i, (role, message) in enumerate(self.messages):
            if message:                
                pre,msg,post = [encode_fn(x) 
                                for x in [role + ":\n", message, seps[i % 2]]]
                inputs += (pre + msg + eos + post)
                
                if role==self.roles[0] or (multi_rounds and i<len(self.messages)-1):
                    labels+= ignore_fn(pre + msg + eos + post)
                else:
                    labels+= (ignore_fn(pre) + msg + eos + ignore_fn(post))
                    
                if i % 2 == 1:
                    tokens = encode_fn("\n\n")
                    inputs += tokens
                    labels += ignore_fn(tokens)
            else:
                pre = encode_fn(role + ":\n")
                inputs += pre
                labels += ignore_fn(pre)
                
        return inputs,labels

    
    elif self.sep_style == SeparatorStyle.PHOENIX:
        inputs = encode_fn(system_prompt)
        labels = ignore_fn(inputs)
        
        for role, message in self.messages:
            if message:
                pre,msg,post = [encode_fn(x) 
                                for x in [role + ": " + "<s>", message, "</s>"]]
                inputs += (pre + msg + eos + post)
                
                if role==self.roles[0] or (multi_rounds and i<len(self.messages)-1):
                    labels+= ignore_fn(pre + msg + eos + post)
                else:
                    labels+= (ignore_fn(pre) + msg + eos + ignore_fn(post))
            else:
                pre = encode_fn(role + ": " + "<s>")
                inputs += pre
                labels += ignore_fn(pre)
                
        return inputs,labels

    
    elif self.sep_style == SeparatorStyle.ROBIN:
        inputs = encode_fn(system_prompt + self.sep)
        labels = ignore_fn(inputs)
        
        for role, message in self.messages:
            if message:
                pre,msg,post = [encode_fn(x) 
                                for x in [role + ":\n", message, self.sep]]
                inputs += (pre + msg + eos + post)
                
                if role==self.roles[0] or (multi_rounds and i<len(self.messages)-1):
                    labels+= ignore_fn(pre + msg + eos + post)
                else:
                    labels+= (ignore_fn(pre) + msg + eos + ignore_fn(post))
            else:
                pre = encode_fn(role + ":\n")
                inputs += pre
                labels += ignore_fn(pre)
        return inputs,labels
    else:
        raise ValueError(f"Invalid style: {self.sep_style}")
        