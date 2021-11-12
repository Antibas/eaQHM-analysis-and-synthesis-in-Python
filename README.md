# extended adaptive Quasi-Harmonic Analysis and Synthesis of Speech in Python

### Author: Panagiotis Antivasis, Undergraduate Student at [Computer Science Department - University of Crete](https://www.csd.uoc.gr/), November 2021.

## Introduction
The following repository is the source code corresponding to the thesis of **Panagiotis Antivasis**, an undergraduate [Computer Science](https://www.csd.uoc.gr/) student at the [University of Crete](https://www.uoc.gr/). 

This code is built upon a speech analysis and synthesis system named [ROBUST FULL-BAND ADAPTIVE SINUSOIDAL ANALYSIS AND SYNTHESIS OF SPEECH, by George P. Kafentzis, Olivier Rosec, Yannis Stylianou](https://www.csd.uoc.gr/~kafentz/Publications/Kafentzis%20G.P.,%20Rosec%20O.,%20and%20Stylianou%20Y.%20Robut%20Adaptive%20Sinusoidal%20Analysis%20and%20Synthesis%20of%20Speech.pdf). The system in the so-called **Extended Adaptive Quasi-Harmonic Model (eaQHM)** and this source code implements it into **Python**.

## eaQHMAnalysisAndSynthesis
**eaQHMAnalysisAndSynthesis** is a function that performs **extended adaptive Quasi-Harmonic Analysis and Synthesis** in a signal and decomposes speech into AM-FM components according to that model. In other words, it receives a *.wav* file and some other optional parameters and produces a [Deterministic component](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.5702&rep=rep1&type=pdf) of the signal by assuming an initially harmonic model by applying an ```f0``` estimation and then iteratively refining it, until the reconstructed signal converges in *quasi-harmonicity*. 

## Prerequisites & Requirements
**Python 3.8.x** and up. It is also highly suggested to use [Spyder](https://www.spyder-ide.org/) environment as the whole code was tested in it. Before you run, make sure to install all requirements by executing:
```Python
pip install -r requirements.txt
```

## How to Run
A *main.py* file is provided, which executes **eaQHMAnalysisAndSynthesis** on a speech signal, whose name is given as an input on the console. The gender of the speaker may also be specified. A wav file *SA19.wav* is provided for you to try it.

What you have to do is:
1. Open *main.py*.
2. Run the code.
3. Give as input the name of the mono *.wav* file to be processed in the console.
4. Specify the gender of the speaker ("male", "female" or other). You may also use "child" as an input.
5. After the program terminates, a *\*filename\*_reconstructed.wav* file will be generated.

Here is an example of the output of the code running *SA19.wav* in [Spyder](https://www.spyder-ide.org/):

![](img/SA19out.JPG)

And here are the plots produced:

![](img/frequencySpec.png)
![](img/frequencySpec2.png)
![](img/timeDom.png)


## Code citation
If you wish to use and cite this code, please use the following:
* Panagiotis Antivasis, eaQHM analysis and synthesis system, (2021), GitHub repository, https://github.com/Antibas/eaQHM-analysis-and-synthesis-in-Python

This code is mostly based on the following publication: 
* G. P. Kafentzis, O. Rosec and Y. Stylianou, "Robust full-band adaptive Sinusoidal analysis and synthesis of speech," 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2014, pp. 6260-6264, doi: 10.1109/ICASSP.2014.6854808.
