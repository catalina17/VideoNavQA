# VideoNavQA

[**VideoNavQA: Bridging the Gap between Visual and Embodied Question Answering**](https://arxiv.org/abs/1908.04950)

[Cătălina Cangea](https://catalinacangea.netlify.com/), [Eugene Belilovsky](http://eugenium.github.io/), [Pietro Liò](https://www.cl.cam.ac.uk/~pl219/), [Aaron Courville](https://mila.quebec/en/person/aaron-courville/)

*BMVC 2019, spotlight talk at ViGIL NeurIPS 2019*

We introduce the **VideoNavQA** task: by removing the navigation and action selection requirements from [Embodied QA](http://embodiedqa.org/), we increase the difficulty of the visual reasoning component via a much larger question space, tackling the sort of complex reasoning questions that make QA tasks challenging. By designing and evaluating several VQA-style models on the dataset, we establish a novel way of evaluating EQA feasibility given existing methods, while highlighting the difficulty of the problem even in the most ideal setting.

## Getting started

### Dataset

The **VideoNavQA** benchmark data can be found [here](https://drive.google.com/drive/folders/1DpEdjmVDMeJZ0ohS_TTp0HAjEbX0fU_m?usp=sharing).

### Dependencies
* Model evaluation:
  * [Faster-RCNN](https://github.com/catalina17/faster-rcnn.pytorch) fork (with VGG-16 pre-trained [weights](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0))
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
