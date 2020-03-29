# VideoNavQA

[**VideoNavQA: Bridging the Gap between Visual and Embodied Question Answering**](https://arxiv.org/abs/1908.04950)  
*[BMVC 2019](https://bmvc2019.org/programme/detailed-programme/), spotlight talk at [ViGIL NeurIPS 2019](https://vigilworkshop.github.io/#schedule)*  
[Cătălina Cangea](https://www.cl.cam.ac.uk/~ccc53/), [Eugene Belilovsky](http://eugenium.github.io/), [Pietro Liò](https://www.cl.cam.ac.uk/~pl219/), [Aaron Courville](https://mila.quebec/en/person/aaron-courville/)

We introduce the **VideoNavQA** task: by removing the navigation and action selection requirements from [Embodied QA](http://embodiedqa.org/), we increase the difficulty of the visual reasoning component via a much larger question space, tackling the sort of complex reasoning questions that make QA tasks challenging. By designing and evaluating several VQA-style models on the dataset, we establish a novel way of evaluating EQA feasibility given existing methods, while highlighting the difficulty of the problem even in the most ideal setting.

![Sample videos](https://github.com/catalina17/VideoNavQA/blob/master/samples/v1.gif) | ![Sample videos](https://github.com/catalina17/VideoNavQA/blob/master/samples/v2.gif) | ![Sample videos](https://github.com/catalina17/VideoNavQA/blob/master/samples/v3.gif)
:---: | :---: | :---:
_'Where is the green rug next to the sofa?'_ | _'Are the computer and the bed the same color?'_ | _'What is the thing next to the tv stand located in the living room?'_

## Getting started

```
$ git clone https://github.com/catalina17/VideoNavQA
$ virtualenv -p python3 videonavqa
$ source videonavqa/bin/activate
$ pip install -r requirements.txt
```

### Dataset

The **VideoNavQA** benchmark data can be found [here](https://drive.google.com/drive/folders/1DpEdjmVDMeJZ0ohS_TTp0HAjEbX0fU_m?usp=sharing). After expanding the archive to a specific directory, please update `BASE_DIR` (declared in `eval/utils.py`) with that path.

![Dataset statistics](https://github.com/catalina17/VideoNavQA/blob/master/samples/dataset%20stats.png)

### Dependencies
* Model evaluation:
  * [Faster-RCNN](https://github.com/catalina17/faster-rcnn.pytorch) fork (with VGG-16 pre-trained [weights](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0))
  * pre-trained object detector for extracting visual features (`OBJ_DETECTOR_PATH` in `eval/utils.py`) should be initialised from [this checkpoint](https://www.dropbox.com/s/o7k0o7d1bwc77du/obj_detect.pt) instead of the one initially provided in the dataset archive - **please make sure to replace the file!**
  
![High-level approach](https://github.com/catalina17/VideoNavQA/blob/master/samples/high%20level.png)

* Data generation tools:
  * [EmbodiedQA](https://github.com/catalina17/EmbodiedQA) fork
  * [House3D](https://github.com/catalina17/House3D) fork
  * SUNCG [dataset](https://sscnet.cs.princeton.edu)
  * SUNCG [toolbox](https://github.com/jjhartmann/SUNCGtoolbox)

## Citation
Please cite us if you use our code or the VideoNavQA benchmark:

```
@article{cangea2019videonavqa,
  title={VideoNavQA: Bridging the Gap between Visual and Embodied Question Answering},
  author={Cangea, C{\u{a}}t{\u{a}}lina and Belilovsky, Eugene and Li{\`o}, Pietro and Courville, Aaron},
  journal={arXiv preprint arXiv:1908.04950},
  year={2019}
}
```
