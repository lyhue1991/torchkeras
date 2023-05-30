import onnxruntime as ort
from pathlib import Path 
import os 

path = Path(__file__)

det_model_path = str(path.parent.parent/"assets"/"ocr_det.onnx")
cls_model_path = str(path.parent.parent/"assets"/"ocr_cls.onnx")
rec_model_path = str(path.parent.parent/"assets"/"ocr_rec.onnx")

class Predictor(object):
    def __init__(self,model_path):
        
        self.session = ort.InferenceSession(model_path)
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.input_tensor = self.session.get_inputs()[0]
        
        #debug 
        #self.model_path = model_path

    def forward(self, x):        
        input_dict = {self.input_tensor.name: x}
        outputs = self.session.run(self.output_names, input_dict)
        
        #debug
        #if 'det' in self.model_path:
        #    print(x.shape) 
        return outputs
        
def create_predictor(args, mode, logger):
    if mode == "det":
        model_dir = args.det_model_dir if args.det_model_dir  else det_model_path
    elif mode == 'cls':
        model_dir = args.cls_model_dir if args.cls_model_dir else cls_model_path
    elif mode == 'rec':
        model_dir = args.rec_model_dir if args.rec_model_dir else rec_model_path
    else:
        raise ValueError('mode={} not in (det,cls,rec)'.format(mode))

    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)
    if not os.path.exists(model_dir):
        raise ValueError("not find model file path {}".format(
            model_dir)) 
    predictor = Predictor(model_path=model_dir)
    
    return predictor