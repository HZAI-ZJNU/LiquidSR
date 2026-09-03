# LiquidSR


## Quickstart
Run the script in ``src`` folder. Before you run the demo, please uncomment the appropriate line in ```demo.sh``` that you want to execute.
```bash
cd src       # You are now in */HNCT/src
sh demo.sh
```

You can find the result images from ```experiment/test/model``` folder.

## How to test LiquidSR
We used [DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf) dataset to train our model.

Unpack the tar file to any place you want. Then, change the ```dir_data``` argument in ```src/option.py``` to the place where DIV2K images are located.

We recommend you to pre-process the images before training. This step will decode all **png** files and save them as binaries. Use ``--ext sep_reset`` argument on your first run. You can skip the decoding part and use saved binaries with ``--ext sep`` argument.

You can train HNCT by yourself. All scripts are provided in the ``src/demo.sh``.

```bash
cd src       # You are now in */HNCT/src
sh demo.sh
```

**Update log**