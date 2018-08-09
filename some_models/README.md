# Example models 

Trained on:
- caps: Multi30k en-de captions
- COCO: MSCOCO captions dataset. Artificial parallel German data translated with a state-of-the art architecture MT model trained with marian-nmt.

# Usage
 - *Translate*: use the OpenNMT way of translationg
 - *Extract the meaning representation matrix*: to load it to python one can use, for instance,
    ```
    import sys
    import torch
    
    PATH_TO_ONMT='/path/to/ATT-ATTbranch'
    sys.path.insert(0, PATH_TO_ONMT)'
    import onmt
    
    def import_checkpoint(fpath):
        checkpoint = torch.load(fpath, map_location=lambda storage, loc: storage)
        return checkpoint
    ```
