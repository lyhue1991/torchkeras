
'''
reference from https://github.com/fastai/fastprogress/
'''
import time
from IPython.display import clear_output, display, HTML


def format_time(t):
    "Format `t` (in seconds) to (h):mm:ss"
    t = int(t)
    h,m,s = t//3600, (t//60)%60, t%60
    if h!= 0: return f'{h}:{m:02d}:{s:02d}'
    else:     return f'{m:02d}:{s:02d}'

def format_number(x):
    if abs(x)<1e-4 or abs(x)>=1e5:
        return "{:.4e}".format(x)
    else:
        return "{:.4f}".format(x)

html_progress_bar_styles = """
<style>
    /* background: */
    progress::-webkit-progress-bar {background-color: #CDCDCD; width: 100%;}
    progress {background-color: #CDCDCD;}

    /* value: */
    progress::-webkit-progress-value {background-color: #00BFFF  !important;}
    progress::-moz-progress-bar {background-color: #00BFFF  !important;}
    progress {color: #00BFFF ;}

    /* optional */
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #000000;
    }
</style>
"""

def html_progress_bar(value, total, label='', postfix='', interrupted=False):
    "Html code for a progress bar `value`/`total` with `label`"
    bar_style = 'progress-bar-interrupted' if interrupted else ''
    val = '' if total is None else f"value='{value}'"
    return f"""
    <div>
      <progress {val} class='{bar_style}' max='{total}' style='width:300px; height:20px; vertical-align: middle;'></progress>
      {label}
      <br>
      {postfix}
    </div>
    """

class ProgressBar:
    update_every,first_its,lt = 0.2,5,'<'
    def __init__(self, gen, total=None, 
                 display=True, comment=''):
        self.gen,self.comment = gen,comment
        self.postfix = ''
        self.total = None if total=='noinfer' else len(gen) if total is None else total
        self.last_v = 0
        self.display = display
        self.last_v = None
        self.update(0)
        
    def update(self, val):
        if self.last_v is None:
            self.on_iter_begin()
            self.last_v = 0
        if val == 0:
            self.start_t = self.last_t = time.time()
            self.pred_t,self.last_v,self.wait_for = None,0,1
            self.update_bar(0)
        elif val <= self.first_its or val >= self.last_v + self.wait_for or (self.total and val >= self.total):
            cur_t = time.time()
            avg_t = (cur_t - self.start_t) / val
            self.wait_for = max(int(self.update_every / (avg_t+1e-8)),1)
            self.pred_t = None if self.total is None else avg_t * self.total
            self.last_v,self.last_t = val,cur_t
            self.update_bar(val)
            if self.total is not None and val >= self.total:
                self.on_iter_end()
                self.last_v = None
                
    def on_iter_begin(self):
        self.html_code = '\n'.join([html_progress_bar(0, self.total, ""), ""])
        display(HTML(html_progress_bar_styles))
        self.out = display(HTML(self.html_code), display_id=True)

    def on_iter_end(self):
        total_time = format_time(time.time() - self.start_t)
        self.comment = f'[{total_time}]' 
        if hasattr(self, 'out'): 
            self.on_update(self.total,self.comment,self.postfix)

    def on_update(self, val, comment='', postfix='', interrupted=False): 
        self.progress = html_progress_bar(val, self.total,comment,postfix,interrupted)
        if self.display: 
            self.out.update(HTML(self.progress))
            
    def on_interrupt(self,msg='interrupted'):
        self.on_update(self.last_v,self.comment+f'[{msg}]',interrupted=True)

    def __iter__(self):
        if self.total != 0: self.update(0)
        try:
            for i,o in enumerate(self.gen):
                if self.total and i >= self.total: break
                yield o
                self.update(i+1)
            if self.total is None and self.last_v is not None:
                self.total = i+1
                self.update(self.total)
        except Exception as e:
            self.on_interrupt()
            raise e

    def update_bar(self, val):
        if self.total == 0:
            warn("Your generator is empty.")
            return self.on_update(0, '100% [0/0]')
        if val ==0:
            self.comment = f'0% [0/{self.total}]'
            return self.on_update(0, self.comment)
        pct = '' if self.total is None else f'{100 * val/self.total:.2f}% '
        tot = '?' if self.total is None else str(self.total)
        elapsed_t = self.last_t - self.start_t
        remaining_t = '?' if self.pred_t is None else format_time(self.pred_t - elapsed_t)
        elapsed_t = format_time(elapsed_t)
        self.comment = f'{pct}[{val}/{tot} {elapsed_t}{self.lt}{remaining_t}]'
        self.on_update(val, self.comment, self.postfix)
    
    def set_postfix(self,**kwargs):
        postfix = ''
        if 'i' in kwargs and 'n' in kwargs:
            from tqdm.std import Bar 
            i,n = kwargs['i'],kwargs['n']
            kwargs.pop('i')
            kwargs.pop('n')
            ratio = i/n
            postfix+=format(Bar(ratio,default_len=20))
            postfix+=f'{100 * i/n:.2f}%'
            postfix+=f' [{i}/{n}] '
        if kwargs:
            postfix+='['
            for i,(key,value) in enumerate(kwargs.items()):
                if isinstance(value,float):
                    postfix = postfix+f'{key}={format_number(value)},'
                else:
                    postfix = postfix+f'{key}={value},'
            postfix = postfix[:-1]+']'
        self.postfix = postfix
        self.on_update(self.last_v,self.comment,self.postfix)