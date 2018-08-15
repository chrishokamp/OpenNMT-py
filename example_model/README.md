# Example model 

Trained EN->DE on the following data:
- caps: Multi30k en-de captions
- COCO: MSCOCO captions dataset. Artificial parallel German data translated with a state-of-the art architecture MT model trained with marian-nmt.

# Usage
 - **Translate**: use the OpenNMT way of translationg
 - **Extract the meaning representation matrix**: call a script, similar to `use_ATT-ATT_embs.py` and call is as one calls the translator script of OpenNMT, for instance,
 
    ```
    python /path/to/ATT-ATTbranch/example_model/use_ATT-ATT_embs.py \
         -model /path/to/ATT-ATTbranch/example_model/COCO+caps_att-heads_12_size_1200_acc_71.76_ppl_5.51_e20.pt \
         -src data/src-test.txt 
    ```
*yes, the `-src` flag is still there, but won't be used, so you can put any string*
